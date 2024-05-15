import numpy as np
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
import wandb
import argparse

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 10000
PATIENCE = 100


def start_sweep(project, sweep_name, regular=False):
    if regular:
        assert "regular" in sweep_name, \
            "Sweep name must contain 'regular' if regular is True."
        sweep_configuration = {
            'method': 'random',
            'name': sweep_name,
            'metric': {'goal': 'maximize', 'name': 'val_loss'},
            'parameters': {
                'lr': {'values': [0.01, 0.001, 0.0001, 0.0001]},
                'weight_decay': {'values': [0.1, 0.01, 0.001]},
                'epsilon': {'values': [1e-8, 1e-5, 1e-2, 1e-1]},
                'batch_size': {'values': [200, 400]},
                'num_hidden_layers': {'values': [1, 2, 3, 4, 5, 6]},
                'hidden_size_multiplier': {'values': [1, 2, 3, 4, 5, 6]},
                'momentum': {'values': [0.5, 0.7, 0.9]},
                'optimizer_type': {'values': ['SGD', 'adamw']},
                'lr_scheduler_type': {'values': [
                    'StepLR', 'ReduceLROnPlateau', 'ExponentialLR', 'None'
                ]},
            }
        }
    else:
        sweep_configuration = {
            'method': 'random',
            'name': sweep_name,
            'metric': {'goal': 'maximize', 'name': 'val_loss'},
            'parameters': {
                'lr': {'values': [0.01, 0.001, 0.0001]},
                'weight_decay': {'values': [0.1, 0.01, 0.001]},
                'epsilon': {'values': [1e-8, 1e-5, 1e-2, 1e-1]},
                'batch_size': {'values': [200, 400]},
                'layer_norm_inverse': {'values': [0]}, # [0, 1]
                'init_gamma': {'values': [0.1, 0.5, 0.75, 0.9, 1.0]},
                'max_gamma': {'values': [1000.0, 10000.0, 100000.0]},
                'anneal_rate': {'values': [0.001, 0.01, 0.1, 0.2, 0.5]},
                'anneal_method': {'values': ['exponential']},
                'num_hidden_layers': {'values': [1, 2, 3, 4, 5, 6]},
                'hidden_size_multiplier': {'values': [1, 2, 3, 4, 5, 6]},
                'momentum': {'values': [0.5, 0.7, 0.9]},
                'optimizer_type': {'values': ['SGD', 'adamw']},
                # 'lr_scheduler_type': {'values': [
                #     'StepLR', 'ReduceLROnPlateau', 'ExponentialLR', 'None'
                # ]},
                'lr_scheduler_type': {'values': ['ReduceLROnPlateau', 'None']},
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
    init_gamma = wandb.config.init_gamma
    # min_gamma = wandb.config.min_gamma
    max_gamma = wandb.config.max_gamma
    anneal_rate = wandb.config.anneal_rate
    anneal_method = wandb.config.anneal_method
    layer_norm_inverse = wandb.config.layer_norm_inverse
    # wp = wandb.config.wp
    num_hidden_layers = wandb.config.num_hidden_layers
    hidden_size_multiplier = wandb.config.hidden_size_multiplier
    momentum = wandb.config.momentum
    optimizer_type = wandb.config.optimizer_type
    lr_scheduler_type = wandb.config.lr_scheduler_type

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

    # Initialize model settings
    init_type = 'ian_normal'
    opt_type = 'greedy'
    activation = 'relu'

    if args.regular:
        # Run with regular layer norm
        norm_type = 'layer'
        model = StrNNDensityEstimator(
            nin=input_size,
            hidden_sizes=hidden_sizes,
            nout=input_size,
            opt_type=opt_type,
            precomputed_masks=None,
            adjacency=adj_mtx,
            activation=activation,
            data_type=data_type,
            init_type=init_type,
            norm_type=norm_type
        )
    else:
        norm_type = 'adaptive_layer'
        model = StrNNDensityEstimator(
            nin=input_size,
            hidden_sizes=hidden_sizes,
            nout=input_size,
            opt_type=opt_type,
            precomputed_masks=None,
            adjacency=adj_mtx,
            activation=activation,
            data_type=data_type,
            init_type=init_type,
            norm_type=norm_type,
            init_gamma=init_gamma,
            # min_gamma=min_gamma,
            max_gamma=max_gamma,
            anneal_rate=anneal_rate,
            anneal_method=anneal_method,
            layer_norm_inverse=layer_norm_inverse
            # wp=wp
        )
    model.to(device)
    # Log init_type, opt_type, activation, norm_type, and data settings
    wandb.log({
        "init_type": init_type,
        "opt_type": opt_type,
        "activation": activation,
        "norm_type": norm_type,
        "data_path": args.data_path,
        "adj_path": args.adj_path,
    })
    print("Initialized model.")

    # Intialize optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum
        )
    elif optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon
        )

    # Initialize lr scheduler 
    if lr_scheduler_type == 'StepLR':
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        # Defaut settings for ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10)
    elif lr_scheduler_type == 'ExponentialLR':
        lr_scheduler = ExponentialLR(optimizer, gamma=0.99)
    elif lr_scheduler_type == 'None':
        lr_scheduler = None
    else:
        raise ValueError("Invalid lr_scheduler_type.")

    # Train model
    best_model_state = train_loop(
        model, optimizer, train_dl, val_dl, MAX_EPOCHS, PATIENCE, lr_scheduler
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--adj_path", type=str)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--sweep_name", type=str, default=None)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--regular", type=int, default=0)
    args = parser.parse_args()
    
    if args.sweep_id is None:
        try:
            # Start new sweep
            sweep_id = start_sweep(args.project, args.sweep_name)
            print(f"Started {args.sweep_name} with id {sweep_id}.")
        except:
            import sys, pdb, traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        sweep_id = args.sweep_id

        # Run wandb agent
        wandb.agent(sweep_id, project=args.project, function=main, count=10)
