import argparse
import os
import yaml

import numpy as np
from scipy.stats import sem

import torch
from torch.utils.data import DataLoader

import wandb

import strnn.models.config_constants as cc
from strnn.models.continuous_flows import AdjacencyModifier
from strnn.models.normalizing_flow import NormalizingFlow

from experiments.train_utils import MODEL_CONSTR_MAP

from utils_mm import load_data

device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_name", type=str)
parser.add_argument("--model_config", type=str)
args = parser.parse_args()


def evaluate_loss(model: NormalizingFlow, test_dl: DataLoader) -> float:
    """Evaluate test negative log likelihood.

    Args:
        model: Model to evaluate.
        test_dl: Dataloader containing test points.

    Returns:
        Mean test NLL across test data points.
    """
    losses = []

    for batch in test_dl:
        z, jac = model(batch)
        loss = model.compute_loss(z, jac)
        losses.append(loss.item())

    test_loss = float(np.mean(losses))

    return test_loss


def get_runs_by_model(wandb_name: str, model_config: str) -> list:
    """Return all runs associated with a particular model config.

    This method is mainly used to select a single model for which multiple
    random initializations have been performed.

    Args:
        wandb_name: Name of wandb project.
        model_config: Name of model config to select.

    Returns:
        List of wandb runs.
    """
    api = wandb.Api()
    runs = api.runs(
        wandb_name,
        filters={"config.model_config": model_config}
    )
    return runs


def get_run_loss(run: wandb.apis.public.Run) -> float:
    """Load model weights from a wandb run and evaluates test loss.

    Args:
        run: The wandb run object containing model config and weights.

    Returns:
        Test negative log likelihood across test data.
    """
    config = run.config

    generator, data = load_data(
        config["dataset_name"],
        config["n_samples"],
        config["split_ratio"],
        config["data_random_seed"]
    )

    test_data = torch.Tensor(data[2]).to(device)
    test_dl: DataLoader = DataLoader(test_data, batch_size=128)

    with open("./config/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
        model_config = model_config[config["model_config"]]

    model_config[cc.INPUT_DIM] = generator.data_dim

    adj_mat = generator.adj_mat
    if config[cc.ADJ_MOD] is not None:
        adj_modifier = AdjacencyModifier(config[cc.ADJ_MOD])
        adj_mat = adj_modifier.modify_adjacency(adj_mat)
    model_config[cc.ADJ] = adj_mat

    model_factory = MODEL_CONSTR_MAP[model_config[cc.BASE_MODEL]](model_config)
    model = model_factory.build_flow().to(device)

    weights = [m for m in run.logged_artifacts() if m.type == "model"][0]
    path = weights.download()
    model_path = os.listdir(path)[0]
    model_weights = torch.load(path + "/" + model_path)
    model.load_state_dict(model_weights)

    loss = evaluate_loss(model, test_dl)

    return loss


def main():
    """Evaluate performance across runs using same config."""
    print(args.model_config, flush=True)
    runs = get_runs_by_model(args.wandb_name, args.model_config)

    test_nll = []

    for run in runs:
        nll = get_run_loss(run)
        test_nll.append(nll)

    mean = np.mean(test_nll)
    se = sem(test_nll)
    print(mean, se)


if __name__ == "__main__":
    main()
