import argparse
from pathlib import Path

import numpy as np
import yaml

import torch

from evaluation import evaluate_intervention, get_nrmse, get_mse
from data.causal_sem import SparseSEM

from utils import dict2namespace
from strnn.models.config_constants import *
from strnn.models.causal_arflow import CausalARFlowTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAREFL_CONFIG_PATH = "./config/baseline.yaml"
STRAF_CONFIG_PATH = "./config/masked.yaml"

parser = argparse.ArgumentParser("Runs flows on causal synthetic dataset.")
parser.add_argument("--n_graph_vars", type=int, required=True)
parser.add_argument("--data_samples", type=int, required=True)
parser.add_argument("--graph_seed", type=int, default=254)

# parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--flow_steps", type=int, default=5)
parser.add_argument("--hidden_width", type=int, default=10)
parser.add_argument("--hidden_depth", type=int, default=4)
parser.add_argument("--n_param_per_var", type=int, default=2)
parser.add_argument("--train_epochs", type=int, default=1000)

parser.add_argument("--eval", type=str, choices=["nrmse", "mse"])
parser.add_argument("--n_eval_points", type=int, default=8)
parser.add_argument("--n_dist_samples", type=int, default=5000)
parser.add_argument("--n_trials", type=int, default=10)
args = parser.parse_args()

print(args, flush=True)


def get_model_config(args: argparse.Namespace, config: dict) -> dict:
    """Update model arch config with command line arguments.

    Args:
        args: Command line arguments.
        config: Model arch config.

    Returns:
        Config dictionary updated with command line arguments.
    """
    assert config[BASE_MODEL] == "ANF"

    hidden_dim = list([args.hidden_width] * args.hidden_depth)
    config[INPUT_DIM] = args.n_graph_vars
    config[COND_HID] = hidden_dim
    config[N_PARAM_PER_VAR] = args.n_param_per_var
    config[FLOW_STEPS] = args.flow_steps

    return config


def main():
    """Generate SEM dataset and evaluate both CAREFL and Causal StrAF."""
    np.random.seed(args.graph_seed)
    torch.manual_seed(2541)

    sem = SparseSEM(args.n_graph_vars)

    if args.eval == "nrmse":
        eval_func = get_nrmse
    else:
        eval_func = get_mse

    # Load non-masked carefl config
    carefl_config_file = open(CAREFL_CONFIG_PATH, "r")
    carefl_config_raw = yaml.load(carefl_config_file, Loader=yaml.FullLoader)
    carefl_config = dict2namespace(carefl_config_raw)
    carefl_config.device = device
    carefl_config.flow.nl = args.flow_steps
    carefl_config.flow.nh = args.hidden_width
    carefl_config.training.epochs = args.train_epochs

    carefl_arch_config = carefl_config_raw["carefl"]
    carefl_arch_config = get_model_config(args, carefl_arch_config)
    carefl_config.arch_config = carefl_arch_config

    # Load masked straf config
    straf_config_file = open(STRAF_CONFIG_PATH, "r")
    straf_config_raw = yaml.load(straf_config_file, Loader=yaml.FullLoader)
    straf_config = dict2namespace(straf_config_raw)
    straf_config.device = device
    straf_config.flow.nl = args.flow_steps
    straf_config.flow.nh = args.hidden_width
    straf_config.training.epochs = args.train_epochs

    straf_arch_config = straf_config_raw["straf"]
    straf_arch_config = get_model_config(args, straf_arch_config)
    straf_config.arch_config = straf_arch_config

    overall_carefl_error = []
    overall_straf_error = []

    for _ in range(args.n_trials):
        dataset = sem.generate_samples(args.data_samples)
        bin_adj_mat = sem.get_adj_mat()

        carefl_config.arch_config[ADJ] = None
        straf_config.arch_config[ADJ] = bin_adj_mat

        carefl = CausalARFlowTrainer(carefl_config)
        straf = CausalARFlowTrainer(straf_config)

        # Train models
        carefl.fit_to_sem(dataset)
        straf.fit_to_sem(dataset)

        # Evaluate
        carefl_true, carefl_pred, _ = evaluate_intervention(
            sem,
            carefl.flow,
            dataset,
            args.n_eval_points,
            args.n_dist_samples
        )

        straf_true, straf_pred, _ = evaluate_intervention(
            sem,
            straf.flow,
            dataset,
            args.n_eval_points,
            args.n_dist_samples
        )

        carefl_err, _ = eval_func(carefl_true, carefl_pred)
        straf_err, _ = eval_func(straf_true, straf_pred)

        carefl_err = np.nanmean(carefl_err)
        straf_err = np.nanmean(straf_err)

        overall_carefl_error.append(carefl_err)
        overall_straf_error.append(straf_err)

    print(np.nanmean(overall_carefl_error), np.nanstd(overall_carefl_error))
    print(np.nanmean(overall_straf_error), np.nanstd(overall_straf_error))

    exp_output = {
        "args": args,
        "carefl_out": overall_carefl_error,
        "straf_out": overall_straf_error,
        "sem": sem
    }

    output_dir_name = "./output/"
    output_dir = Path("./output/")
    output_dir.mkdir(parents=False, exist_ok=True)

    out_fn = "linadd{}_{}hid_{}obs_{}runs_{}"

    out_fn = out_fn.format(
        args.n_graph_vars,
        args.hidden_width,
        args.data_samples,
        args.n_trials,
        args.eval
    )

    torch.save(exp_output, output_dir_name + out_fn + ".pt")


if __name__ == "__main__":
    main()
