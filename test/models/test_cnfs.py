import pytest

import numpy as np

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


@pytest.mark.parametrize("ode_type", ["strnn", "weilbach", "fully_connected"])
def test_cnf_forward(ode_type):
    config[ODENET_LIN] = ode_type
    factory = ContinuousFlowFactory(config)
    cnf = factory.build_cnf()

    assert isinstance(cnf, ContinuousFlow)

    out, jac = cnf(rand_in)
    assert torch.is_tensor(jac)
    assert jac.shape == (batch_size, 1)

    inv, jac = cnf.invert(out)
    assert torch.is_tensor(jac)
    assert jac.shape == (batch_size, 1)

    assert torch.allclose(rand_in, inv, rtol=1e-2, atol=1e-2)
