import argparse
import yaml

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from experiments.train_utils import train_loop, check_wandb_run, persist_wandb
from experiments.train_utils import MODEL_CONSTR_MAP

from utils_mm import load_data

import strnn.models.config_constants as cc
from strnn.models.continuous_flows import AdjacencyModifier

import wandb


device = torch.device("cuda:0")

parser = argparse.ArgumentParser("Runs flows on multimodal synthetic dataset.")
parser.add_argument("--dataset_name", type=str, default="multimodal")
parser.add_argument("--data_random_seed", type=int, default=2547)
parser.add_argument("--n_samples", type=int, default=1000)
parser.add_argument("--split_ratio", type=eval, default=[0.6, 0.2, 0.2])

parser.add_argument("--adj_mod", type=eval)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--lr", type=float)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--scheduler", type=str)

parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--flow_steps", type=int)
parser.add_argument("--hidden_width", type=int)
parser.add_argument("--hidden_depth", type=int)
parser.add_argument("--model_seed", type=int)

parser.add_argument("--umnn_hidden_width", type=int)
parser.add_argument("--umnn_hidden_depth", type=int)
parser.add_argument("--n_param_per_var", type=int)

parser.add_argument("--wandb_name", type=str)
parser.add_argument("--persist", type=eval, default=False)
parser.add_argument("--wandb_check", type=eval, default=False)
args = parser.parse_args()


def parse_args_model(args: argparse.Namespace, config: dict) -> dict:
    """Update model config with command line arguments.

    Args:
        args: Command line arguments.
        config: Model config.

    Returns:
        Config dictionary updated with command line arguments.
    """
    new_config = {}

    if args.hidden_width is not None and args.hidden_depth is not None:
        hidden_dim = tuple([args.hidden_width] * args.hidden_depth)

        if config[cc.BASE_MODEL] == "ANF":
            new_config[cc.COND_HID] = hidden_dim
        elif config[cc.BASE_MODEL] == "CNF":
            new_config[cc.ODENET_HID] = hidden_dim
        else:
            raise ValueError("Unknown base model type.")

    umnn_w_set = args.umnn_hidden_width is not None
    umnn_d_set = args.umnn_hidden_depth is not None

    if umnn_w_set and umnn_d_set:
        umnn_hid_dim = tuple([args.umnn_hidden_width] * args.umnn_hidden_depth)
        new_config[cc.UMNN_INT_HID] = umnn_hid_dim

    if args.n_param_per_var is not None:
        new_config[cc.N_PARAM_PER_VAR] = args.n_param_per_var

    if args.flow_steps is not None:
        new_config[cc.FLOW_STEPS] = args.flow_steps

    config = config | new_config
    return config


def main():
    """Training script for StrAF and other normalizing flows."""
    generator, data = load_data(args.dataset_name, args.n_samples,
                                args.split_ratio, args.data_random_seed)

    train_data = torch.Tensor(data[0]).to(device)
    val_data = torch.Tensor(data[1]).to(device)

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    with open("./config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_config = config[args.model_config]

    model_config[cc.INPUT_DIM] = generator.data_dim
    model_config = parse_args_model(args, model_config)

    adj_mat = generator.adj_mat
    if args.adj_mod is not None:
        adj_modifier = AdjacencyModifier(args.adj_mod)
        adj_mat = adj_modifier.modify_adjacency(adj_mat)
        model_config[cc.ADJ_MOD] = args.adj_mod
    model_config[cc.ADJ] = adj_mat

    model_factory = MODEL_CONSTR_MAP[model_config[cc.BASE_MODEL]](model_config)

    # Fixed seed prior to initialization. Useful for generating CIs.
    if args.model_seed is not None:
        np.random.seed(args.model_seed)
        torch.random.manual_seed(args.model_seed)

    model = model_factory.build_flow().to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)

    if args.scheduler is None or args.scheduler == "fixed":
        args.scheduler = "fixed"
        scheduler = None
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    else:
        raise ValueError("Unknown scheduler.")

    train_args = {
        "lr": args.lr,
        "max_epoch": args.max_epochs,
        "patience": args.patience,
        "use_wandb": True,
        "scheduler": args.scheduler
    }

    if args.wandb_name:
        # Reset adjacency to ground truth for persistence. Any modifiers
        # are stored and can be recreated.
        model_config[cc.ADJ] = generator.adj_mat

        all_config = vars(args)
        all_config.update(model_config)

        if args.wandb_check and check_wandb_run(all_config, args.wandb_name):
            print("Run with same config is running or complete.")
            return

        run = wandb.init(
            project=args.wandb_name,
            config=all_config
        )

    best_model_state = train_loop(
        model,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
        train_args
    )

    if args.wandb_name and args.persist:
        persist_wandb(run, best_model_state)

    run.finish()


if __name__ == "__main__":
    main()
