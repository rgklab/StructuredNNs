import argparse
import yaml

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
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
    # Load experiment configs
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)
        experiment_config = configs[args.experiment_name]

    # Load data
    dataset_name = experiment_config["dataset_name"]
    adj_mtx_name = experiment_config["adj_mtx_name"]
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(
        dataset_name, adj_mtx_name
    )
    input_size = len(train_data[0])
    experiment_config["input_size"] = input_size

    # Specify hidden layer sizes
    num_hidden_layers = experiment_config["num_hidden_layers"]
    hidden_size_mults = []
    for i in range(1, 6):
        hidden_size_mults.append(
            experiment_config[f"hidden_size_multiplier_{i}"]
        )
    hidden_sizes = [h * input_size for h in hidden_size_mults]
    hidden_sizes = tuple(hidden_sizes[:num_hidden_layers])
    assert isinstance(hidden_sizes[0], int)

    # Make dataloaders
    batch_size = experiment_config["batch_size"]
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Intialize model (fix random seed if necessary)
    if args.model_seed is not None:
        np.random.seed(args.model_seed)
        torch.random.manual_seed(args.model_seed)

    if "binary" in dataset_name:
        data_type = "binary"
        output_size = input_size
    elif "gaussian" in dataset_name:
        data_type = "gaussian"
        output_size = 2 * input_size
    else:
        raise ValueError("Data type must be binary or Gaussian!")

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
        init_type=experiment_config["init_type"],
        norm_type=experiment_config["norm_type"],
        gamma=experiment_config["gamma"],
        wp=experiment_config["wp"]
    )
    model.to(device)

    # Initialize optimizer
    optimizer_name = experiment_config["optimizer"]
    if optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=experiment_config["learning_rate"],
            eps=experiment_config["epsilon"],
            weight_decay=experiment_config["weight_decay"]
        )
    else:
        raise ValueError(f"{optimizer_name} is not a valid optimizer!")

    run = wandb.init(
        project=args.wandb_name,
        config=experiment_config
    )

    best_model_state = train_loop(
        model,
        optimizer,
        train_dl,
        val_dl,
        experiment_config["max_epochs"],
        experiment_config["patience"]
    )


if __name__ == "__main__":
    main()