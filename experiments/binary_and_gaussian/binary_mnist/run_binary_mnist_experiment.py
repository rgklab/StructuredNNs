import argparse

import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from binary_gaussian_mnist_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

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

    final_val_losses_ian = []
    final_val_losses_kaiming = []
    for num_layers in range(1, 10):
        hidden_sizes = tuple(h * input_size for h in hidden_size_mults[:num_layers])

        model_ian = StrNNDensityEstimator(
            nin=input_size,
            hidden_sizes=hidden_sizes,
            nout=output_size,
            opt_type=experiment_config["opt_type"],
            opt_args={},
            precomputed_masks=None,
            adjacency=adj_mtx,
            activation=experiment_config["activation"],
            data_type=data_type,
            init_type='ian_uniform'
        )
        model_ian.to(device)

        optimizer_ian = AdamW(
            model_ian.parameters(),
            lr=experiment_config["learning_rate"],
            eps=experiment_config["epsilon"],
            weight_decay=experiment_config["weight_decay"]
        )

        results_ian = train_loop(
            model_ian,
            optimizer_ian,
            train_dl,
            val_dl,
            experiment_config["max_epochs"],
            experiment_config["patience"],
        )
        final_val_losses_ian.append(results_ian['val_losses_per_epoch'][-1])

        model_kaiming = StrNNDensityEstimator(
            nin=input_size,
            hidden_sizes=hidden_sizes,
            nout=output_size,
            opt_type=experiment_config["opt_type"],
            opt_args={},
            precomputed_masks=None,
            adjacency=adj_mtx,
            activation=experiment_config["activation"],
            data_type=data_type,
            init_type='kaiming_uniform'
        )
        model_kaiming.to(device)

        optimizer = AdamW(
            model_kaiming.parameters(),
            lr=experiment_config["learning_rate"],
            eps=experiment_config["epsilon"],
            weight_decay=experiment_config["weight_decay"]
        )

        results_kaiming = train_loop(
            model_kaiming,
            optimizer,
            train_dl,
            val_dl,
            experiment_config["max_epochs"],
            experiment_config["patience"]
        )
        final_val_losses_kaiming.append(results_kaiming['val_losses_per_epoch'][-1])

    # After the loop, plot the final validation losses for both initialization methods
    fig, ax = plt.subplots()
    ax.plot(range(1, 10), final_val_losses_ian, marker='o', linestyle='-', label='Ian Init')
    ax.plot(range(1, 10), final_val_losses_kaiming, marker='o', linestyle='-', label='Kaiming Init')
    ax.set_title('Final Validation Loss Over Layers (d100)')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Final Validation Loss')
    ax.legend()
    wandb.log({'Final Validation Loss Over Layers (Ian vs Kaiming)': wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()
