import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Tuple

DSTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def split_dataset(
    data: np.ndarray,
    split_ratio: tuple[float, float, float]
) -> DSTuple:
    """Split data into train, validation, and test splits.

    Args:
        data: Dataset to split. Assumes first dimension is sample dimension.
        split_ratio: Ratio of train, validation, and test splits.
    Return:
        Tuple of train, validation, test data splits.
    """
    test_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])

    temp_set, train_set = train_test_split(data, test_size=split_ratio[0])
    val_set, test_set = train_test_split(temp_set, test_size=test_ratio)

    return train_set, val_set, test_set


def standardize_data(data: np.ndarray):
    """Standardize data.

    Args:
        data: Data of dimension (n_samples, n_features).
    Returns:
        Standardized data.
    """
    return StandardScaler().fit_transform(data)
