import argparse
import yaml

import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data.data_utils import split_dataset, standardize_data
from data.make_adj_mtx import generate_adj_mat_uniform
from data.synthetic_multimodal import SyntheticMultimodalDataset

from strnn.models.discrete_flows import AutoregressiveFlowFactory
from strnn.models.continuous_flows import ContinuousFlowFactory
from strnn.models import NormalizingFlowLearner

from strnn.models.config_constants import ADJ, INPUT_DIM
from strnn.models.config_constants import BASE_MODEL, FLOW_STEPS, ODENET_HID

from data.data_utils import DSTuple


parser = argparse.ArgumentParser("Runs flows on multimodal synthetic dataset.")
parser.add_argument("--dataset_name", type=str, default="multimodal")
parser.add_argument("--data_random_seed", type=int, default=2547)
parser.add_argument("--n_samples", type=int, default=5000)
parser.add_argument("--split_ratio", type=eval, default=[0.6, 0.2, 0.2])

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=-1)

parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--flow_steps", type=int, default=3)
parser.add_argument("--hidden_dim", type=eval, default=(100, 100))
parser.add_argument("--model_seed", type=int, default=2547)

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

    generator = SyntheticMultimodalDataset(**data_config)
    data = generator.generate_samples(n_samples, random_seed)
    standard_data = standardize_data(data)
    train, val, test = split_dataset(standard_data, split_ratio)

    return generator, (train, val, test)


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
    model_config[ADJ] = generator.adj_mat
    model_config[FLOW_STEPS] = args.flow_steps
    model_config[ODENET_HID] = args.hidden_dim

    model_factory = MODEL_CONSTR_MAP[model_config[BASE_MODEL]](model_config)

    # Fixed seed prior to initialization. Useful for generating CIs.
    np.random.seed(args.model_seed)
    torch.random.manual_seed(args.model_seed)

    model = model_factory.build_flow()

    learner = NormalizingFlowLearner(model, args.lr)

    train_args = {"max_epochs": args.max_epochs}

    if args.wandb_name:
        all_config = vars(args)
        all_config.update(model_config)

        wandb_logger = WandbLogger(project=args.wandb_name)
        wandb_logger.experiment.config.update(all_config)
        train_args["logger"] = wandb_logger

    if args.patience != -1:
        early_stopping = EarlyStopping("val_loss", patience=args.patience)
        train_args["callbacks"] = [early_stopping]

    trainer = pl.Trainer(**train_args)

    trainer.fit(model=learner,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
