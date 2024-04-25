import numpy as np

from data.data_utils import split_dataset
from data.data_utils import standardize_data


def test_split_dataset():
    """Test train/test splitting function."""
    ratio = (0.6, 0.2, 0.2)
    data = np.ones((100, 1))

    train, val, test = split_dataset(data, ratio)

    assert len(train) == data.shape[0] * ratio[0]
    assert len(val) == data.shape[0] * ratio[1]
    assert len(test) == data.shape[0] * ratio[2]


def test_standardize_data():
    """Test standardization function."""
    data = np.random.normal(100, 0.1, size=(5000, 1))

    standardized_data = standardize_data(data)

    assert np.allclose(np.mean(standardized_data), 0)
    assert np.allclose(np.std(standardized_data), 1)
