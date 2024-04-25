import pytest

import torch

from strnn.models.discrete_flows.normalizers import MonotonicNormalizer
from strnn.models.discrete_flows.normalizers import AffineNormalizer


batch_size = 32
input_dim = 10


def test_affine_normalizer():
    """Test dimensions and invertibility of affine normalizer."""
    affine = AffineNormalizer()

    data = torch.randn(batch_size, input_dim)
    params = torch.randn(batch_size, input_dim, 2)

    z, jac = affine.forward(data, params)

    assert z.shape == data.shape
    assert jac.shape == data.shape

    x = affine.invert(z, params)

    assert x.shape == z.shape

    # Test invertibility
    assert torch.allclose(x, data, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("solver", ["CC", "CCParallel"])
def test_monotonic_normalizer(solver):
    """Test dimensions and invertibility of monotonic normalizer."""
    config = {
        "integrand_hidden": [32, 32],
        "n_param_per_dim": 8,
        "nb_steps": 20,
        "solver": solver,
    }

    monotonic = MonotonicNormalizer(**config)

    data = torch.randn(batch_size, input_dim)
    params = torch.randn(batch_size, input_dim, config["n_param_per_dim"])

    z, jac = monotonic.forward(data, params)

    assert z.shape == data.shape
    assert jac.shape == data.shape

    x = monotonic.invert(z, params)

    assert x.shape == z.shape

    # Test invertibility
    assert torch.allclose(x, data, rtol=1e-2, atol=1e-2)
