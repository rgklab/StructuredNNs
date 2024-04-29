import argparse

import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from binary_gaussian_train_utils import load_data_and_adj_mtx
from strnn.models.strNN import MaskedLinear
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_name", type=str, default="multimodal")
parser.add_argument("--data_seed", type=int, default=2547)
parser.add_argument("--scheduler", type=str, default="plateau")
parser.add_argument("--model_seed", type=int, default=2647)
parser.add_argument("--wandb_name", type=str)

args = parser.parse_args()


def compute_kai_layer_variances(model):
    layer_variances = []
    for layer_idx, layer in enumerate(model.net_list[:-1]):
        if isinstance(layer, MaskedLinear):
            layer_variance = torch.var(layer.weight.data).item()
            layer_variances.append(layer_variance)
    return layer_variances


def compute_ian_layer_variances(model):
    layer_variances = []
    for layer_idx, layer in enumerate(model.net_list[:-1]):
        if isinstance(layer, MaskedLinear):
            masked_weights = layer.weight.data * layer.mask
            non_zero_elements = masked_weights[masked_weights != 0]
            if non_zero_elements.numel() > 0:
                layer_variance = torch.var(non_zero_elements).item() * (
                            non_zero_elements.numel() / masked_weights.numel())
                layer_variances.append(layer_variance)
    return layer_variances


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

    hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 10)]

    data_type = "binary" if "binary" in dataset_name else "gaussian"
    output_size = input_size if data_type == "binary" else 2 * input_size

    run = wandb.init(project="strnn_init", entity="strnn-init", config=experiment_config, reinit=True)

    hidden_sizes = tuple(h * input_size for h in hidden_size_mults[:6])

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
        init_type="ian_uniform"
    ).to(device)

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
    ).to(device)

    # Compute layer variances for both models
    variances_ian = compute_ian_layer_variances(model_ian)
    variances_kaiming = compute_kai_layer_variances(model_kaiming)
    print("ian_variances", variances_ian)
    print("kai_variances", variances_kaiming)

    # Expected variance for each layer
    expected_variances = [2 / h for h in [input_size] + hidden_sizes[:5]]
    print("expected kai variance", expected_variances)

    for layer_idx in range(5):
        plt.figure(figsize=(5, 5))
        plt.scatter(['Ian Init'], [variances_ian[layer_idx]], color='blue', label='Ian Init')
        plt.scatter(['Kaiming Init'], [variances_kaiming[layer_idx]], color='green', label='Kaiming Init')
        plt.scatter(['Expected Kaiming'], [expected_variances[layer_idx]], color='red', label='Expected Kaiming')
        plt.ylabel('Variance')
        plt.ylim(0, 0.025)
        plt.title(f'Layer {layer_idx + 1} Variance Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_filename = f'layer_{layer_idx + 1}_variances.png'
        plt.savefig(plot_filename)
        plt.close()

        wandb.log({f"Layer Variance Comparison": wandb.Image(plot_filename)})


if __name__ == "__main__":
    main()
