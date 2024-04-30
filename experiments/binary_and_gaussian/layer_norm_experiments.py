import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
import argparse

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 5000
PATIENCE = 20


def start_sweep(project, sweep_name):
    sweep_configuration = {
        'method': 'random',
        'name': sweep_name,
        'metric': {'goal': 'maximize', 'name': 'val_loss'},
        'parameters': {
            'lr': {'max': 0.1, 'min': 0.001},
            'weight_decay': {'values': [0.1, 0.01, 0.001]},
            'epsilon': {'values': [1e-8, 1e-5, 1e-2, 1e-1]},
            'batch_size': {'values': [100, 200, 400]},
            'gamma': {'values': [0.1, 0.25, 0.5, 0.75, 1.0]},
            'wp': {'values': [0, 0.25, 0.5, 0.75, 1.0]},
            'num_hidden_layers': {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            'hidden_size_multiplier': {'values': [1, 2, 3, 4, 5, 6]},
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    wandb.agent(sweep_id, project=project, function=main, count=1)
    return sweep_id


def main():
    """
    Main function to be run by wandb agent
    """
    # Get experiment configs from wandb
    run = wandb.init()
    lr = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    epsilon = wandb.config.epsilon
    batch_size = wandb.config.batch_size
    gamma = wandb.config.gamma
    wp = wandb.config.wp
    num_hidden_layers = wandb.config.num_hidden_layers
    hidden_size_multiplier = wandb.config.hidden_size_multiplier

    # Fix random seed if necessary
    if args.model_seed is not None:
        np.random.seed(args.model_seed)
        torch.random.manual_seed(args.model_seed)

    # Load data and adjacency matrix
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(
        args.data_path, args.adj_path, load_test=False
    )
    input_size = len(train_data[0])
    wandb.log({"input_size": input_size})

    # Make dataloaders
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    print("Dataloaders created.")

    # Specify hidden layer sizes
    hidden_sizes = [hidden_size_multiplier * input_size] * num_hidden_layers

    # Specify binary or Gaussian data
    if 'binary' in args.data_path:
        data_type = "binary"
    elif 'gaussian' in args.data_path:
        data_type = "gaussian"
    else:
        raise ValueError("Data type must be binary or Gaussian!")

    # Initialize model
    model = StrNNDensityEstimator(
        nin=input_size,
        hidden_sizes=hidden_sizes,
        nout=input_size,
        opt_type="greedy",
        precomputed_masks=None,
        adjacency=adj_mtx,
        activation="relu",
        data_type=data_type,
        init_type="ian_normal",
        norm_type="adaptive_layer",
        gamma=gamma,
        wp=wp
    )
    model.to(device)
    print("Initialized model.")

    # Intialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=epsilon
    )

    # Train model
    best_model_state = train_loop(
        model, optimizer, train_dl, val_dl, MAX_EPOCHS, PATIENCE
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--adj_path", type=str)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--sweep_name", type=str, default=None)
    parser.add_argument("--model_seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.sweep_id is None:
        # Start new sweep
        sweep_id = start_sweep(args.project, args.sweep_name)
        print(f"Started {args.sweep_name} with id {sweep_id}.")
    else:
        sweep_id = args.sweep_id

        # Run wandb agent
        wandb.agent(sweep_id, project=args.project, function=main, count=50)
