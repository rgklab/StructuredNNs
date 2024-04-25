import pytest

import numpy as np
import torch

from strnn.models.config_constants import *
from strnn.models.continuous_flows.continuous_flow import *

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

    OPT_TYPE: "greedy",
    OPT_ARGS: {},

    ODENET_HID: [50, 50, 50],
    ODENET_ACT: "tanh",

    ODENET_DIV_FN: "brute_force",
    ODE_TIME: 0.5,
    ODE_TRAIN_T: True,

    ODE_SOLVER_TYPE: "dopri5",
    ODE_SOLVER_ATOL: 1e-3,
    ODE_SOLVER_RTOL: 1e-5,
    ODE_SOLVER_STEP: None,
}

batch_size = 32
rand_in = torch.randn(batch_size, adj_mat.shape[0])


def test_factory():
    """Test initialization of CNF factory."""
    config[ODENET_LIN] = "strnn"
    factory = ContinuousFlowFactory(config)

    assert hasattr(factory, "config")


@pytest.mark.parametrize("ode_type", ["strnn", "weilbach", "fully_connected"])
def test_cnf_forward(ode_type):
    """Test dimensional correctness of CNF forward pass."""
    config[ODENET_LIN] = ode_type
    factory = ContinuousFlowFactory(config)
    cnf = factory.build_flow()

    assert hasattr(cnf, "config")
    assert isinstance(cnf, ContinuousFlow)

    out, jac = cnf(rand_in)
    assert torch.is_tensor(jac)
    assert jac.shape == (batch_size, 1)

    inv = cnf.invert(out)

    assert torch.allclose(rand_in, inv, rtol=1e-2, atol=1e-2)


def test_adj_modifier_diag():
    """Test main diagonal adjacency modification."""
    true_adj_mat = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 1]
    ])
    modifiers = ["main_diagonal"]

    adj_mod = AdjacencyModifier(modifiers)
    mod_adj_mat = adj_mod.modify_adjacency(adj_mat)
    assert np.all(mod_adj_mat == true_adj_mat)


def test_adj_modifier_reflect():
    """Test reflection adjacency modification."""
    true_adj_mat = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    modifiers = ["reflect"]

    adj_mod = AdjacencyModifier(modifiers)
    mod_adj_mat = adj_mod.modify_adjacency(adj_mat)
    assert np.all(mod_adj_mat == true_adj_mat)


def test_adj_modifier_reflect_diag():
    """Test combination of diagonal and reflection adjacency modification."""
    true_adj_mat = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1]
    ])
    modifiers = ["main_diagonal", "reflect"]

    adj_mod = AdjacencyModifier(modifiers)
    mod_adj_mat = adj_mod.modify_adjacency(adj_mat)
    assert np.all(mod_adj_mat == true_adj_mat)
