import numpy as np

import torch
from torchdiffeq import odeint

from strnn.models.continuous_flows.odenets import WeilbachSparseLinear
from strnn.models.continuous_flows.odenets import WeilbachSparseODENet
from strnn.models.continuous_flows.odenets import FCODEnet
from strnn.models.continuous_flows.odenets import StrODENet


A = np.array(
    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 1, 1, 0]]
)

input_dim = A.shape[0]
hidden_dim = [25, 25, 25]
act_type = "tanh"

opt_type = "greedy"
opt_args = {}

batch_size = 32
rand_in = torch.randn(batch_size, input_dim)
rand_t = torch.randn(1, dtype=torch.float)
int_times = torch.arange(0, 5, dtype=torch.float)


def test_weilbach_lin():
    layer = WeilbachSparseLinear(input_dim, input_dim, torch.Tensor(A))

    output = layer(rand_t, rand_in)
    assert output.shape == rand_in.shape

    int_out = odeint(layer, rand_in, int_times).permute(1, 2, 0)
    assert int_out.shape == (batch_size, input_dim, len(int_times))


def test_weilbach_odenet():
    odenet = WeilbachSparseODENet(input_dim, len(hidden_dim), act_type, A)

    output = odenet(rand_t, rand_in)
    assert output.shape == rand_in.shape

    int_out = odeint(odenet, rand_in, int_times).permute(1, 2, 0)
    assert int_out.shape == (batch_size, input_dim, len(int_times))


def test_fcodenet():
    odenet = FCODEnet(input_dim, hidden_dim, act_type)
    output = odenet(rand_t, rand_in)
    assert output.shape == rand_in.shape

    int_out = odeint(odenet, rand_in, int_times).permute(1, 2, 0)
    assert int_out.shape == (batch_size, input_dim, len(int_times))


def test_strode():
    odenet = StrODENet(
        input_dim,
        hidden_dim,
        input_dim,
        opt_type,
        opt_args,
        adjacency=A,
        activation=act_type
    )
    output = odenet(rand_t, rand_in)
    assert output.shape == rand_in.shape

    int_out = odeint(odenet, rand_in, int_times).permute(1, 2, 0)
    assert int_out.shape == (batch_size, input_dim, len(int_times))
