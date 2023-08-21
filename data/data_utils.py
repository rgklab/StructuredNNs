import numpy as np


def standardize_data(data: np.ndarray):
    """Standardizes data.

    Args:
        data: Data of dimension (n_samples, n_features).
    Returns:
        Standardized data.
    """
    norm_data = np.zeros((data.shape[0], data.shape[1]))
    for i, row in enumerate(data):
        norm_data[i] = (row - row.mean()) / row.std()
    return norm_data
