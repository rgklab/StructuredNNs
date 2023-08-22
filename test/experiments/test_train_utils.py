import numpy as np

from experiments.train_utils import compute_sample_mmd


def test_compute_sample_mmd_same():
    n_samples = 5000
    input_dim = 10

    mean = 0
    std = 1

    X = np.random.normal(mean, std, size=(n_samples, input_dim))
    Y = np.random.normal(mean, std, size=(n_samples, input_dim))

    mmd = compute_sample_mmd(X, Y, gamma=0.1)
    assert np.allclose(mmd, 0, atol=1e-1)


def test_compute_sample_mmd_different():
    n_samples = 5000
    input_dim = 10

    mean_1 = 0
    std_1 = 1

    mean_2 = 10
    std_2 = 5

    X = np.random.normal(mean_1, std_1, size=(n_samples, input_dim))
    Y = np.random.normal(mean_2, std_2, size=(n_samples, input_dim))

    mmd = compute_sample_mmd(X, Y, gamma=0.1)
    assert not np.allclose(mmd, 0, atol=1e-1)
