import argparse
import yaml

import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from data.data_utils import split_dataset, standardize_data
from data.make_adj_mtx import generate_adj_mat_uniform
from data.synthetic_multimodal import SyntheticMultimodalDataset

from strnn.models.discrete_flows import AutoregressiveFlowFactory
from strnn.models.continuous_flows import AdjacencyModifier
from strnn.models.continuous_flows import ContinuousFlowFactory
from strnn.models import NormalizingFlowLearner
from strnn.models.config_constants import *

from experiments.train_utils import CallbackComputeMMD

from data.data_utils import DSTuple

parser = argparse.ArgumentParser("Runs flows on multimodal synthetic dataset.")
parser.add_argument("--dataset_name", type=str, default="multimodal")
parser.add_argument("--data_random_seed", type=int, default=2547)
parser.add_argument("--n_samples", type=int, default=3000)
parser.add_argument("--split_ratio", type=eval, default=[0.6, 0.2, 0.2])

parser.add_argument("--adj_mod", type=eval)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--mmd_samples", type=int)
parser.add_argument("--mmd_gamma", type=float, default=0.1)
parser.add_argument("--scheduler", type=str, default="fixed")

parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--flow_steps", type=int, default=3)
parser.add_argument("--hidden_width", type=int, default=300)
parser.add_argument("--hidden_depth", type=int, default=2)
parser.add_argument("--model_seed", type=int, default=2541)

parser.add_argument("--umnn_hidden_width", type=int, default=100)
parser.add_argument("--umnn_hidden_depth", type=int, default=3)
parser.add_argument("--n_param_per_var", type=int, default=30)

parser.add_argument("--wandb_name", type=str)
args = parser.parse_args()


A_GEN_FN_KEY = "adj_mat_gen_fn"
A_GEN_FN_MAP = {"uniform": generate_adj_mat_uniform}


MODEL_CONSTR_MAP = {
    "CNF": ContinuousFlowFactory,
    "ANF": AutoregressiveFlowFactory,
}


def load_data(
    dataset_name: str,
    n_samples: int,
    split_ratio: tuple[float, float, float],
    random_seed: int,
    config_path: str = "./data_config.yaml"
) -> tuple[SyntheticMultimodalDataset, DSTuple]:
    """Generates data samples, applies preprocessing and data splits.

    Args:
        dataset_name: Name of dataset in config.
        n_sample: Number of samples to generate.
        split_ratio: Ratio of train / val / test splits.
        random_seed: Random seed used to draw samples.
        config_path: Path to data config file.

    Returns:
        Data generator, and tuple containing arrays of train / val / test data.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        data_config = config[dataset_name]
        data_config[A_GEN_FN_KEY] = A_GEN_FN_MAP[data_config[A_GEN_FN_KEY]]

    np.random.seed(random_seed)
    generator = SyntheticMultimodalDataset(**data_config)
    data = generator.generate_samples(n_samples, random_seed)
    standard_data = standardize_data(data)
    train, val, test = split_dataset(standard_data, split_ratio)

    return generator, (train, val, test)


def parse_args_model(args: argparse.Namespace, config: dict) -> dict:
    """Updates model config with command line arguments.

    Args:
        args: Command line arguments.
        config: Model config.

    Returns:
        Config dictionary updated with command line arguments.
    """
    hidden_dim = tuple([args.hidden_width] * args.hidden_depth)

    if config[BASE_MODEL] == "ANF":
        config[COND_HID] = hidden_dim

        umnn_hid_dim = tuple([args.umnn_hidden_width] * args.umnn_hidden_depth)
        config[UMNN_INT_HID] = umnn_hid_dim

        config[N_PARAM_PER_VAR] = args.n_param_per_var

    elif config[BASE_MODEL] == "CNF":
        config[ODENET_HID] = hidden_dim

    else:
        raise ValueError("Unknown base model type.")

    config[FLOW_STEPS] = args.flow_steps

    return config


def main():
    generator, data = load_data(args.dataset_name, args.n_samples,
                                args.split_ratio, args.data_random_seed)

    train_data = torch.Tensor(data[0])
    val_data = torch.Tensor(data[1])

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    with open("./model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_config = config[args.model_config]

    model_config[INPUT_DIM] = generator.data_dim
    model_config = parse_args_model(args, model_config)

    adj_mat = generator.adj_mat
    if args.adj_mod is not None:
        adj_modifier = AdjacencyModifier(args.adj_mod)
        adj_mat = adj_modifier.modify_adjacency(adj_mat)
    model_config[ADJ] = adj_mat

    adj_mat = generator.adj_mat
    if args.adj_mod is not None:
        adj_modifier = AdjacencyModifier(args.adj_mod)
        adj_mat = adj_modifier.modify_adjacency(adj_mat)
    model_config[ADJ] = adj_mat

    model_factory = MODEL_CONSTR_MAP[model_config[BASE_MODEL]](model_config)

    # Fixed seed prior to initialization. Useful for generating CIs.
    np.random.seed(args.model_seed)
    torch.random.manual_seed(args.model_seed)

    model = model_factory.build_flow()

    learner = NormalizingFlowLearner(model, args.lr, args.scheduler)

    train_args = {
        "max_epochs": args.max_epochs,
        "callbacks": [],
        "enable_checkpointing": False,
        "deterministic": True
    }

    if args.wandb_name:
        # Reset adjacency to ground truth for persistence. Any modifiers
        # are stored and can be recreated.
        model_config[ADJ] = generator.adj_mat

        all_config = vars(args)
        all_config.update(model_config)

        wandb_logger = WandbLogger(project=args.wandb_name)

        if rank_zero_only.rank == 0:
            wandb_logger.experiment.config.update(all_config)

        train_args["logger"] = wandb_logger

    if args.patience != -1:
        early_stopping = EarlyStopping("val_loss", patience=args.patience)
        train_args["callbacks"].append(early_stopping)

    if args.mmd_samples > 0:
        mmd_callback = CallbackComputeMMD(args.mmd_samples, args.mmd_gamma)
        train_args["callbacks"].append(mmd_callback)

    trainer = pl.Trainer(**train_args)

    trainer.fit(model=learner,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
