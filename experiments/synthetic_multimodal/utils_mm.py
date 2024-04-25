import yaml

import numpy as np

from data.data_utils import split_dataset, standardize_data, DSTuple
from data.make_adj_mtx import generate_adj_mat_uniform
from data.synthetic_multimodal import SyntheticMultimodalDataset


A_GEN_FN_KEY = "adj_mat_gen_fn"
A_GEN_FN_MAP = {"uniform": generate_adj_mat_uniform}


def load_data(
    dataset_name: str,
    n_samples: int,
    split_ratio: tuple[float, float, float],
    random_seed: int,
    config_path: str = "./config/data_config.yaml"
) -> tuple[SyntheticMultimodalDataset, DSTuple]:
    """Generate data samples, applies preprocessing and data splits.

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
    data = generator.generate_samples(n_samples)
    standard_data = standardize_data(data)
    train, val, test = split_dataset(standard_data, split_ratio)

    return generator, (train, val, test)
