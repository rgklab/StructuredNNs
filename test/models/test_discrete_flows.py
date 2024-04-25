import pytest

import torch
import numpy as np

from strnn.models.discrete_flows import AutoregressiveFlowFactory
from strnn.models.config_constants import *


batch_size = 32

adj_mat = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0]
])

config = {
    INPUT_DIM: adj_mat.shape[0],
    ADJ: adj_mat,
    FLOW_STEPS: 3,
    FLOW_PERMUTE: False,
    COND_HID: [50, 50, 50],
    COND_ACT: "tanh",
}

affine_config = {
    NORM_TYPE: "affine",
    N_PARAM_PER_VAR: 2,
}

umnn_config = {
    NORM_TYPE: "umnn",
    UMNN_INT_HID: [50, 50],
    UMNN_INT_SOLVER: "CC",
    UMNN_INT_STEP: 25,
    N_PARAM_PER_VAR: 10,
}

strnn_config = {
    COND_TYPE: "strnn",
    OPT_ARGS: {},
    OPT_TYPE: "greedy"
}

gnf_config = {
    COND_TYPE: "gnf",
    GNF_HOT: True
}

strnn_affine_config = config.copy()
strnn_affine_config.update(strnn_config)
strnn_affine_config.update(affine_config)

strnn_umnn_config = config.copy()
strnn_umnn_config.update(strnn_config)
strnn_umnn_config.update(umnn_config)

gnf_affine_config = config.copy()
gnf_affine_config.update(gnf_config)
gnf_affine_config.update(affine_config)

gnf_umnn_config = config.copy()
gnf_umnn_config.update(gnf_config)
gnf_umnn_config.update(umnn_config)

all_config = [
    strnn_affine_config,
    strnn_umnn_config,
    gnf_affine_config,
    gnf_umnn_config,
]


def test_affine_incorrect_dim():
    """Check that incorrect dimension raises error."""
    err_config = strnn_affine_config.copy()
    err_config[N_PARAM_PER_VAR] = 4

    with pytest.raises(AssertionError):
        AutoregressiveFlowFactory(err_config)


@pytest.mark.parametrize("config", all_config)
def test_flow_init(config):
    """Test dimensions and invertibility of autoregressive flow."""
    factory = AutoregressiveFlowFactory(config)
    flow = factory.build_flow()
    assert hasattr(flow, "config")

    test_in = torch.randn(batch_size, config[INPUT_DIM])

    z, jac = flow.forward(test_in)
    assert z.shape == test_in.shape
    assert jac.shape == (test_in.shape[0],)

    x_inv = flow.invert(z)
    assert x_inv.shape == test_in.shape

    assert torch.allclose(test_in, x_inv, atol=5e-1, rtol=5e-1)


@pytest.mark.parametrize("n_steps", [3, 4])
def test_flow_permute(n_steps):
    """Test autoregressive flows permute latents correctly."""
    permute_config = strnn_affine_config.copy()
    permute_config[FLOW_PERMUTE] = True
    permute_config[FLOW_STEPS] = n_steps

    factory = AutoregressiveFlowFactory(permute_config)
    flow = factory.build_flow()

    test_in = torch.randn(batch_size, config[INPUT_DIM])

    # Output should be permuted
    z, _ = flow.forward(test_in)

    x_inv = flow.invert(z)
    assert x_inv.shape == test_in.shape

    assert torch.allclose(test_in, x_inv, atol=1, rtol=1)
