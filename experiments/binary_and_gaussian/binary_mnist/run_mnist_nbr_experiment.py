import argparse
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from binary_gaussian_mnist_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Runs StrNN on synthetic dataset.")
parser.add_argument("--experiments", nargs='+', help="List of experiments with different nbr_sizes", required=True)
parser.add_argument("--data_seed", type=int, default=2547)
parser.add_argument("--scheduler", type=str, default="plateau")
parser.add_argument("--model_seed", type=int, default=2647)
parser.add_argument("--wandb_name", type=str)

args = parser.parse_args()


def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    val_losses_per_experiment = defaultdict(list)

    for experiment_name in args.experiments:
        experiment_config = configs[experiment_name]

        dataset_name = experiment_config["dataset_name"]
        adj_mtx_name = experiment_config["adj_mtx_name"]
        nbr_size = experiment_config["nbr_size"]  # Assuming nbr_size is defined in each experiment's config
        train_data, val_data, adj_mtx = load_data_and_adj_mtx(dataset_name, adj_mtx_name)
        input_size = len(train_data[0])
        experiment_config["input_size"] = input_size

        batch_size = experiment_config["batch_size"]
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 10)]

        data_type = "binary"
        output_size = input_size if data_type == "binary" else 2 * input_size

        run = wandb.init(project=args.wandb_name, config=experiment_config, reinit=True)

        final_val_losses_kaiming = []
        for num_layers in range(1, 10):
            hidden_sizes = tuple(h * input_size for h in hidden_size_mults[:num_layers])

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

        val_losses_per_experiment[f'nbr_size = {nbr_size}'] = final_val_losses_kaiming
        wandb.finish()

    # After the loop, plot the final validation losses for each nbr_size
    fig, ax = plt.subplots()
    for label, val_losses in val_losses_per_experiment.items():
        ax.plot(range(1, 10), val_losses, marker='o', linestyle='-', label=label)
    ax.set_title('Binary MNIST Label 3')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Final Validation Loss')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
