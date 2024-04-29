import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_names", type=str, nargs='+', help="List of experiment names")
parser.add_argument("--wandb_name", type=str)
parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for each experiment to compute confidence intervals")
args = parser.parse_args()

def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    sparsity_levels = [0.975, 0.95, 0.8, 0.7, 0.5, 0.0]
    run = wandb.init(project=args.wandb_name, reinit=True)

    final_val_losses_ian = {level: [] for level in sparsity_levels}
    final_val_losses_kaiming = {level: [] for level in sparsity_levels}

    for i, experiment_name in enumerate(args.experiment_names):
        for trial in range(args.num_trials):
            experiment_config = configs[experiment_name]
            train_data, val_data, adj_mtx = load_data_and_adj_mtx(experiment_config["dataset_name"], experiment_config["adj_mtx_name"])
            input_size = len(train_data[0])
            batch_size = experiment_config["batch_size"]
            num_hidden_layers = experiment_config["num_hidden_layers"]
            hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 4)]
            hidden_sizes = [h * input_size for h in hidden_size_mults]
            hidden_sizes = tuple(hidden_sizes[:num_hidden_layers])
            data_type = "binary"

            model_ian = StrNNDensityEstimator(
                nin=input_size,
                hidden_sizes=hidden_sizes,
                nout=input_size,
                opt_type=experiment_config["opt_type"],
                opt_args={},
                precomputed_masks=None,
                adjacency=adj_mtx,
                activation=experiment_config["activation"],
                data_type=data_type,
                init_type='ian_uniform'
            ).to(device)
            optimizer_ian = AdamW(model_ian.parameters(),
                                  lr=experiment_config["learning_rate"],
                                  eps=experiment_config["epsilon"],
                                  weight_decay=experiment_config["weight_decay"])
            results_ian = train_loop(model_ian,
                                     optimizer_ian,
                                     DataLoader(train_data, batch_size=batch_size, shuffle=True),
                                     DataLoader(val_data, batch_size=batch_size, shuffle=False),
                                     experiment_config["max_epochs"],
                                     experiment_config["patience"])
            final_val_losses_ian[sparsity_levels[i]].append(results_ian['val_losses_per_epoch'][-1])

            model_kaiming = StrNNDensityEstimator(
                nin=input_size,
                hidden_sizes=hidden_sizes,
                nout=input_size,
                opt_type=experiment_config["opt_type"],
                opt_args={},
                precomputed_masks=None,
                adjacency=adj_mtx,
                activation=experiment_config["activation"],
                data_type=data_type,
                init_type='kaiming_uniform'
            ).to(device)

            optimizer_kaiming = AdamW(model_kaiming.parameters(),
                                      lr=experiment_config["learning_rate"],
                                      eps=experiment_config["epsilon"],
                                      weight_decay=experiment_config["weight_decay"])
            results_kaiming = train_loop(model_kaiming,
                                         optimizer_kaiming,
                                         DataLoader(train_data, batch_size=batch_size, shuffle=True),
                                         DataLoader(val_data, batch_size=batch_size, shuffle=False),
                                         experiment_config["max_epochs"],
                                         experiment_config["patience"])
            final_val_losses_kaiming[sparsity_levels[i]].append(results_kaiming['val_losses_per_epoch'][-1])

    # Compute mean validation losses
    mean_ian = np.array([np.mean(final_val_losses_ian[level]) for level in sparsity_levels])
    mean_kaiming = np.array([np.mean(final_val_losses_kaiming[level]) for level in sparsity_levels])

    # Calculate percentage difference relative to Kaiming
    percentage_diff = 100 * (mean_kaiming - mean_ian) / mean_kaiming

    # Plotting percentage differences
    fig, ax = plt.subplots()
    ax.plot(sparsity_levels, percentage_diff, label='Percentage Difference (Kaiming - Ian)', color='purple')
    ax.set_title('Percentage Difference in Validation Loss (Ian vs Kaiming)')
    ax.set_xlabel('Sparsity Level')
    ax.set_ylabel('Percentage Difference (%)')
    ax.legend()
    plt.show()
    wandb.log({'Percentage Difference in Validation Loss Over Sparsity Levels': wandb.Image(fig)})
    fig.savefig('percentage_diff_validation_loss_vs_sparsity.png')

if __name__ == "__main__":
    main()