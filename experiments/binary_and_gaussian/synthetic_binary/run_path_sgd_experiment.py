import argparse
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from path_sgd_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_name", type=str, default="multimodal")
parser.add_argument("--data_seed", type=int, default=2547)
parser.add_argument("--scheduler", type=str, default="plateau")
parser.add_argument("--model_seed", type=int, default=2647)
parser.add_argument("--wandb_name", type=str)

args = parser.parse_args()


def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)
    experiment_config = configs[args.experiment_name]

    dataset_name = experiment_config["dataset_name"]
    adj_mtx_name = experiment_config["adj_mtx_name"]
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(dataset_name, adj_mtx_name)
    input_size = len(train_data[0])
    experiment_config["input_size"] = input_size

    batch_size = experiment_config["batch_size"]
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 6)]

    data_type = "binary" if "binary" in dataset_name else "gaussian"
    output_size = input_size if data_type == "binary" else 2 * input_size

    run = wandb.init(project=args.wandb_name, config=experiment_config, reinit=True)

    final_val_losses = []
    for num_layers in range(1, 10):
        hidden_sizes = tuple(h * input_size for h in hidden_size_mults[:num_layers])
        model = StrNNDensityEstimator(
            nin=input_size,
            hidden_sizes=hidden_sizes,
            nout=output_size,
            opt_type=experiment_config["opt_type"],
            opt_args={},
            precomputed_masks=None,
            adjacency=adj_mtx,
            activation=experiment_config["activation"],
            data_type=data_type,
        )
        model.to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=experiment_config["learning_rate"],
            eps=experiment_config["epsilon"],
            weight_decay=experiment_config["weight_decay"]
        )

        results = train_loop(
            model=model,
            optimizer=optimizer,
            train_dl=train_dl,
            val_dl=val_dl,
            max_epoch=experiment_config["max_epochs"],
            patience=experiment_config["patience"],
            input_dim=input_size
        )

        # train/validation losses
        fig, ax = plt.subplots()
        ax.plot(range(len(results['train_losses_per_epoch'])), results['train_losses_per_epoch'], label='Train Loss')
        ax.plot(range(len(results['val_losses_per_epoch'])), results['val_losses_per_epoch'], label='Validation Loss')
        ax.set_title(f'Train/Validation Loss for {num_layers} Layers')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        wandb.log({f'Train/Validation Loss for {num_layers} Layers': wandb.Image(fig)})
        plt.close(fig)

        final_val_losses.append(results['val_losses_per_epoch'][-1])

    # final validation losses over layers
    fig, ax = plt.subplots()
    ax.plot(range(1, 10), final_val_losses, marker='o', linestyle='-')
    ax.set_title('Final Validation Loss Over Layers')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Final Validation Loss')
    wandb.log({'Final Validation Loss Over Layers': wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()
