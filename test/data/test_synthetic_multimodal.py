import numpy as np

from data.make_adj_mtx import generate_adj_mat_uniform
from data.synthetic_multimodal import SyntheticMultimodalDataset


def test_synthetic_multimodal():
    """Test dimensional correctness of synthetic mm data generation."""
    data_dim = 10
    n_samples = 1000

    dataset = SyntheticMultimodalDataset(
        data_dim=data_dim,
        adj_mat_gen_fn=generate_adj_mat_uniform,
        adj_mat_gen_args={"threshold": 0.5},
        n_modes=3,
        w_range=(-3, 3),
        mean_range=(-5, 5),
        std_range=(1, 2))

    samples = dataset.generate_samples(n_samples)

    assert samples.shape == (n_samples, data_dim)


def test_synthetic_multimodal_sample_seed_unfixed():
    """Test that synthetic data generation results in unique data."""
    data_dim = 10
    n_samples = 1000
    seed_1 = 2547
    seed_2 = 2541

    dataset = SyntheticMultimodalDataset(
        data_dim=data_dim,
        adj_mat_gen_fn=generate_adj_mat_uniform,
        adj_mat_gen_args={"threshold": 0.5},
        n_modes=3,
        w_range=(-3, 3),
        mean_range=(-5, 5),
        std_range=(1, 2))

    samples_1 = dataset.generate_samples(n_samples, seed_1)
    samples_2 = dataset.generate_samples(n_samples, seed_2)

    assert not np.all(samples_1 == samples_2)


def test_synthetic_multimodal_sample_seed_fixed():
    """Test synthetic data generation reproducibility with fixed seed."""
    data_dim = 10
    n_samples = 1000
    seed = 2547

    dataset = SyntheticMultimodalDataset(
        data_dim=data_dim,
        adj_mat_gen_fn=generate_adj_mat_uniform,
        adj_mat_gen_args={"threshold": 0.5},
        n_modes=3,
        w_range=(-3, 3),
        mean_range=(-5, 5),
        std_range=(1, 2))

    dataset.initialize_distributions()

    samples_1 = dataset.generate_samples(n_samples, seed)
    samples_2 = dataset.generate_samples(n_samples, seed)

    assert np.all(samples_1 == samples_2)


def test_synthetic_multimodal_param_initialize():
    """Test that loading data generator from config is reproducible."""
    data_dim = 10
    n_samples = 1000
    seed = 2547

    dataset = SyntheticMultimodalDataset(
        data_dim=data_dim,
        adj_mat_gen_fn=generate_adj_mat_uniform,
        adj_mat_gen_args={"threshold": 0.5},
        n_modes=3,
        w_range=(-3, 3),
        mean_range=(-5, 5),
        std_range=(1, 2))

    adj_mat_1 = dataset.adj_mat
    w_mat_1 = dataset.w_mat

    dataset.initialize_distributions()
    params_1 = dataset.dist_params

    samples_1 = dataset.generate_samples(n_samples, seed)

    dataset_new = SyntheticMultimodalDataset.init_from_param(params_1)
    samples_2 = dataset.generate_samples(n_samples, seed)

    assert np.all(adj_mat_1 == dataset_new.adj_mat)
    assert np.all(w_mat_1 == dataset_new.w_mat)
    assert np.all(samples_1 == samples_2)
