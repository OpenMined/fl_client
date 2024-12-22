import importlib.util
import shutil
from datetime import datetime
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from syftbox.lib import Client
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from utils import (
    ProjectStateCols,
    add_public_write_permission,
    create_project_state,
    get_app_private_data,
    read_json,
    search_files,
    update_project_state,
)

DATASET_FILE_PATTERN = r"^mnist_label_[0-9]\.pt$"


# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass


def init_client_app(client: Client) -> None:
    """
    Creates the `fl_client` app in the `api_data` folder
    with the following structure:
    ```
    api_data
    └── fl_client
            └── request
            └── running
    ```
    """
    fl_client = client.api_data("fl_client")

    for folder in ["request", "running", "done"]:
        fl_client_folder = fl_client / folder
        fl_client_folder.mkdir(parents=True, exist_ok=True)

    # Give public write permission to the request folder
    add_public_write_permission(client, fl_client / "request")

    # We additionally create a private folder for the client to place the datasets
    private_folder_path = get_app_private_data(client, "fl_client")
    private_folder_path.mkdir(parents=True, exist_ok=True)


def init_shared_dirs(client: Client, proj_folder: Path) -> None:
    """Creates the shared directories for the project.
    These directories are shared between the client and the aggregator.
    a. round_weights
    b. agg_weights
    c. state
    """

    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    round_weights_folder.mkdir(parents=True, exist_ok=True)
    agg_weights_folder.mkdir(parents=True, exist_ok=True)

    # Give public write permission to the round_weights and agg_weights folder
    add_public_write_permission(client, agg_weights_folder)
    
    # Create a state folder to track progress of the project
    # and give public read permission to the state folder for the aggregator
    create_project_state(client, proj_folder)


def load_model_class(model_path: Path) -> type:
    """Load the model class from the model architecture file"""
    model_class_name = "FLModel"
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    model_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arch)
    model_class = getattr(model_arch, model_class_name)

    return model_class



def train_model(proj_folder: Path, round_num: int, dataset_path_files: list[Path]) -> None:
    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    fl_config_path = proj_folder / "fl_config.json"
    fl_config = read_json(fl_config_path)

    # Load model and aggregator weights
    model_class = load_model_class(proj_folder / fl_config["model_arch"])
    model: nn.Module = model_class()
    agg_weights_file = agg_weights_folder / f"agg_model_round_{round_num - 1}.pt"
    model.load_state_dict(torch.load(agg_weights_file, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=fl_config["learning_rate"])

    # Load datasets
    all_datasets = []
    for dataset_path_file in dataset_path_files:
        images, labels = torch.load(str(dataset_path_file), weights_only=True)
        dataset = TensorDataset(images, labels)
        all_datasets.append(dataset)

    combined_dataset = ConcatDataset(all_datasets)

    # Save dataset size to JSON
    dataset_size = len(combined_dataset)
    dataset_size_file = proj_folder / "dataset_size.json"
    with open(dataset_size_file, "w") as f:
        json.dump({"dataset_size": dataset_size}, f, indent=4)

    # Create DataLoader
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    # Logging
    logs_folder_path = proj_folder / "logs"
    logs_folder_path.mkdir(parents=True, exist_ok=True)

    # 1) Text log file
    output_logs_path = logs_folder_path / f"training_logs_round_{round_num}.txt"
    log_file = open(str(output_logs_path), "w")

    # 2) JSON loss log file
    training_loss_file = logs_folder_path / f"training_loss_round_{round_num}.json"
    if training_loss_file.exists():
        with open(training_loss_file, "r") as f:
            training_loss_data = json.load(f)
    else:
        training_loss_data = []

    start_msg = f"[{datetime.now().isoformat()}] Starting training...\n"
    log_file.write(start_msg)
    log_file.flush()

    update_project_state(
        proj_folder,
        ProjectStateCols.MODEL_TRAIN_PROGRESS,
        f"Training Started for Round {round_num}",
    )

    # Training loop
    for epoch in range(fl_config["epoch"]):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        # Write to text file
        log_msg = f"[{datetime.now().isoformat()}] Epoch {epoch + 1:02d}: Loss = {avg_loss:.6f}\n"
        log_file.write(log_msg)
        log_file.flush()

        # Also track loss in JSON
        training_loss_data.append({
            "epoch": epoch + 1,
            "loss": avg_loss
        })

        update_project_state(
            proj_folder,
            ProjectStateCols.MODEL_TRAIN_PROGRESS,
            f"Training InProgress for Round {round_num} (Curr Epoch: {epoch + 1}/{fl_config['epoch']})",
        )

    # Serialize the model
    output_model_path = round_weights_folder / f"trained_model_round_{round_num}.pt"
    torch.save(model.state_dict(), str(output_model_path))

    # Final log
    final_msg = f"[{datetime.now().isoformat()}] Training completed. Final loss: {avg_loss:.6f}\n"
    log_file.write(final_msg)
    log_file.close()

    update_project_state(
        proj_folder,
        ProjectStateCols.MODEL_TRAIN_PROGRESS,
        f"Training Completed for Round {round_num}",
    )

    # Save the updated JSON loss file
    with open(training_loss_file, "w") as f:
        json.dump(training_loss_data, f, indent=4)




def shift_project_to_done_folder(
    client: Client, proj_folder: Path, total_rounds: int
) -> None:
    """
    Moves the project to the `done` folder
    a. Create a directory in the `done` folder with the same name as the project
    b. moves the agg weights and round weights to the done folder
    c. delete the project folder from the running folder
    """
    done_proj_folder = client.api_data(f"fl_client/done/{proj_folder.name}")
    done_proj_folder.mkdir(parents=True, exist_ok=True)

    # Move the agg weights and round weights folder to the done project folder
    shutil.move(proj_folder / "agg_weights", done_proj_folder)
    shutil.move(proj_folder / "round_weights", done_proj_folder)

    # Delete the project folder from the running folder
    print(f"Deleting project folder from the running folder: {proj_folder.resolve()}")
    shutil.rmtree(proj_folder)


def get_train_datasets(client: Client, proj_folder: Path) -> list[Path]:
    # Check if datasets are present in the private folder
    dataset_path = get_app_private_data(client, "fl_client")
    dataset_path_files = search_files(DATASET_FILE_PATTERN, dataset_path)

    if len(dataset_path_files) == 0:
        raise StateNotReady(
            f"⛔ No dataset found in private folder: {dataset_path.resolve()}"
            "Skipping training "
        )

    update_project_state(proj_folder, ProjectStateCols.DATASET_ADDED, True)

    return dataset_path_files


def has_project_completed(client: Client, proj_folder: Path, total_rounds: int) -> bool:
    """Check if the project has completed model training all the rounds."""

    agg_weights_folder = proj_folder / "agg_weights"
    agg_weights_cnt = len(list(agg_weights_folder.glob("*.pt")))

    # Aggregated weights folder include round weights + init seed weight
    # If aggregated weights for all rounds are present, then the project is completed
    if agg_weights_cnt == total_rounds + 1:
        print(f"FL project {proj_folder.name} has completed all the rounds ✅ 🚀")
        shift_project_to_done_folder(client, proj_folder, total_rounds)
        return True

    return False


def perform_model_training(
    client: Client,
    proj_folder: Path,
    dataset_files: list[Path],
) -> None:
    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    fl_config_path = proj_folder / "fl_config.json"
    fl_config = read_json(fl_config_path)
    total_rounds = fl_config["rounds"]
    current_round = len(list(round_weights_folder.iterdir())) + 1

    # Check if project completed...
    if has_project_completed(client, proj_folder, total_rounds):
        return

    # Check aggregator weights...
    agg_weights_file = agg_weights_folder / f"agg_model_round_{current_round - 1}.pt"
    if not agg_weights_file.is_file():
        raise StateNotReady(
            f"Aggregator has not sent the weights for the round {current_round}"
        )

    # Train the model
    train_model(proj_folder, current_round, dataset_files)

    # Share the trained model
    trained_model_file = round_weights_folder / f"trained_model_round_{current_round}.pt"
    share_model_to_aggregator(
        client,
        fl_config["aggregator"],
        proj_folder,
        trained_model_file,
    )

    # Share dataset size info
    share_dataset_info_to_aggregator(
        client,
        fl_config["aggregator"],
        proj_folder,
    )
    # Copy the training_loss JSON to the public folder
    training_loss_file = proj_folder / "logs" / f"training_loss_round_{current_round}.json"
    if training_loss_file.exists():
        print("Yes it does exists")
        # Public folder for this client under the project name
        public_folder = client.my_datasite / "public" / "fl" / proj_folder.name
        public_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy the file to public so it’s always accessible
        shutil.copy(training_loss_file, public_folder)


def share_model_to_aggregator(
    client: Client,
    aggregator_email: str,
    proj_folder: Path,
    model_file: Path,
) -> None:
    """Shares the trained model to the aggregator."""
    fl_aggregator_app_path = (
        client.datasites / f"{aggregator_email}/api_data/fl_aggregator"
    )
    fl_aggregator_running_folder = fl_aggregator_app_path / "running" / proj_folder.name
    fl_aggregator_client_path = (
        fl_aggregator_running_folder / "fl_clients" / client.email
    )

    # Copy the trained model to the aggregator's client folder
    shutil.copy(model_file, fl_aggregator_client_path)

def share_dataset_info_to_aggregator(
    client: Client,
    aggregator_email: str,
    proj_folder: Path,
) -> None:
    """Shares the dataset size info to the aggregator."""
    fl_aggregator_app_path = (
        client.datasites / f"{aggregator_email}/api_data/fl_aggregator"
    )
    fl_aggregator_running_folder = fl_aggregator_app_path / "running" / proj_folder.name
    fl_aggregator_client_path = (
        fl_aggregator_running_folder / "fl_clients" / client.email
    )

    dataset_size_file = proj_folder / "dataset_size.json"
    if dataset_size_file.exists():
        shutil.copy(dataset_size_file, fl_aggregator_client_path)
    else:
        raise ValueError("dataset_size.json not found on the client side.")



def _advance_fl_project(client: Client, proj_folder: Path) -> None:
    """
    Iterate over all the project folder, it will try to advance its state.
    1. Ensure the project has init directories (like round_weights, agg_weights)
    2. Has the aggregate sent the weights for the current round x (in the agg_weights folder)
    b. The client trains the model on the given round  and places the trained model in the round_weights folder
    c. It sends the trained model to the aggregator.
    d. repeat a until all round completes
    """

    try:
        # Init the shared directories for the project
        init_shared_dirs(client, proj_folder)

        # Retrieve datasets from the private folder if available
        dataset_files = get_train_datasets(client, proj_folder)

        # Train the model for the given FL round
        perform_model_training(client, proj_folder, dataset_files)

    except StateNotReady as e:
        print(e)
        return


def advance_fl_projects(client: Client) -> None:
    """
    Iterates over the `running` folder and tries to advance the FL projects
    """
    running_folder = client.api_data("fl_client") / "running"
    for proj_folder in running_folder.iterdir():
        if proj_folder.is_dir():
            proj_name = proj_folder.name
            print(
                f"Advancing FL project {proj_name} -> proj_folder: {proj_folder.resolve()}"
            )
            _advance_fl_project(client, proj_folder)


def start_app():
    client = Client.load()

    # Step 1: Init the FL Aggregator App
    init_client_app(client)

    # Step 2: Advance the FL Projects.
    # Iterates over the running folder and tries to advance the FL project
    advance_fl_projects(client)


if __name__ == "__main__":
    start_app()
