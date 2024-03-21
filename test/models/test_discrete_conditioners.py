import numpy as np

import torch

from strnn.models.discrete_flows.conditioners import GNFConditioner
from strnn.models.discrete_flows.conditioners import StrNNConditioner


adj_mat = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0]
])

config = {
    "input_dim": 4,
    "hidden_dim": [25, 25],
    "n_out_param": 8,
    "act_type": "relu",
    "adj_mat": adj_mat
}

batch_size = 32


def test_gnf_conditioner_no_hot():
    """Test dimension correctness of GNF that doesn't append OHE to input."""
    test_in = torch.randn(batch_size, config["input_dim"])

    gnf_config = config.copy()
    gnf_config["hot_encoding"] = False
    gnf_conditioner = GNFConditioner(**gnf_config)

    out = gnf_conditioner(test_in)

    target_shape = (batch_size, config["input_dim"], config["n_out_param"])
    assert out.shape == target_shape


def test_gnf_conditioner_hot():
    """Test dimension correctness of GNF that appends OHE to input."""
    test_in = torch.randn(batch_size, config["input_dim"])

    gnf_config = config.copy()
    gnf_config["hot_encoding"] = True
    gnf_conditioner = GNFConditioner(**gnf_config)

    out = gnf_conditioner(test_in)

    target_shape = (batch_size, config["input_dim"], config["n_out_param"])
    assert out.shape == target_shape


def test_straf_conditioner():
    """Test dimension correctness of StrNN conditioner."""
    test_in = torch.randn(batch_size, config["input_dim"])

    straf_config = config.copy()
    straf_config["opt_type"] = "greedy"
    straf_config["opt_args"] = {}

    straf_conditioner = StrNNConditioner(**straf_config)
    out = straf_conditioner(test_in)

    target_shape = (batch_size, config["input_dim"], config["n_out_param"])
    assert out.shape == target_shape
