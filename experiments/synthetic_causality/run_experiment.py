import argparse
import sys

import numpy as np
import yaml

import torch

from evaluation import evaluate_intervention, get_nrmse, get_mse
from data.causal_sem import SparseSEM

from utils import dict2namespace
from strnn.models.config_constants import *
from strnn.models.causal_arflow import CausalARFlow

BL_CONFIG_PATH = "./config/baseline.yaml"
FM_CONFIG_PATH = "./config/masked.yaml"

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
    """Updates model arch config with command line arguments.

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

    np.random.seed(args.graph_seed)
    torch.manual_seed(2541)

    sem = SparseSEM(args.n_graph_vars)

    overall_bl_error = []
    overall_fm_error = []

    if args.eval == "nrmse":
        eval_func = get_nrmse
    else:
        eval_func = get_mse

    # Load non-masked carefl config
    bl_config_file = open(BL_CONFIG_PATH, "r")
    bl_config_raw = yaml.load(bl_config_file, Loader=yaml.FullLoader)
    bl_config = dict2namespace(bl_config_raw)
    bl_config.device = torch.device('cpu')
    bl_config.flow.nl = args.flow_steps
    bl_config.flow.nh = args.hidden_width
    bl_config.training.epochs = args.train_epochs
    bl_config.training.verbose = False

    bl_arch_config = bl_config_raw["carefl"]
    bl_arch_config = get_model_config(args, bl_arch_config)
    bl_config.arch_config = bl_arch_config

    # Load masked straf config
    fm_config_file = open(FM_CONFIG_PATH, "r")
    fm_config_raw = yaml.load(fm_config_file, Loader=yaml.FullLoader)
    fm_config = dict2namespace(fm_config_raw)
    fm_config.device = torch.device('cpu')
    fm_config.flow.nl = args.flow_steps
    fm_config.flow.nh = args.hidden_width
    fm_config.training.epochs = args.train_epochs
    fm_config.training.verbose = False

    fm_arch_config = fm_config_raw["straf"]
    fm_arch_config = get_model_config(args, fm_arch_config)
    fm_config.arch_config = fm_arch_config

    for _ in range(args.n_trials):
        # Vary sample generation
        dataset = sem.generate_samples(args.data_samples)
        bin_adj_mat = sem.get_adj_mat()

        bl_config.arch_config[ADJ] = None
        fm_config.arch_config[ADJ] = bin_adj_mat

        carefl = CausalARFlow(bl_config)
        straf = CausalARFlow(fm_config)

        # Train models
        carefl.fit_to_sem(dataset)
        straf.fit_to_sem(dataset)

        # Evaluate
        bl_true, bl_pred, _ = evaluate_intervention(sem, carefl, dataset,
                                                    args.n_eval_points,
                                                    args.n_dist_samples)
        fm_true, fm_pred, _ = evaluate_intervention(sem, straf, dataset,
                                                    args.n_eval_points,
                                                    args.n_dist_samples)

        bl_err, _ = eval_func(bl_true, bl_pred)
        fm_err, _ = eval_func(fm_true, fm_pred)

        bl_err = np.nanmean(bl_err)
        fm_err = np.nanmean(fm_err)

        overall_bl_error.append(bl_err)
        overall_fm_error.append(fm_err)

    print(np.nanmean(overall_bl_error), np.nanstd(overall_bl_error), flush=True)
    print(np.nanmean(overall_fm_error), np.nanstd(overall_fm_error), flush=True)

    exp_output = {
        "args": args,
        "bl_out": overall_bl_error,
        "fm_out": overall_fm_error,
        "sem": sem
    }

    output_dir = "./output/"
    out_fn = "linadd{}_{}hid_{}obs_{}runs_{}"

    out_fn = out_fn.format(args.n_graph_vars, args.hidden_width, args.data_samples,
                        args.n_trials, args.eval)

    torch.save(exp_output, output_dir + out_fn + ".pt")


if __name__ == "__main__":
    main()