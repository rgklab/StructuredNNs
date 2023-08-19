import copy

import torch
import torch.nn as nn

from ..strNN import StrNN
from ..model_utils import NONLINEARITIES


class WeilbachSparseLinear(nn.Module):
    """Concatenated squash linear layer which encodes predefined adjacency.

    See: http://proceedings.mlr.press/v108/weilbach20a/weilbach20a.pdf.
    Implementation taken from https://github.com/plai-group/daphne with
    minor changes to adapt to PyTorch Lightning.
    """
    def __init__(self, dim_in: int, dim_out: int, adj_mat: torch.Tensor):
        super(WeilbachSparseLinear, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.adj_mat = adj_mat

        _weight_mask = torch.zeros([dim_out, dim_in])
        _weight_mask[:adj_mat.shape[0], :adj_mat.shape[1]] = adj_mat
        self._weight_mask = _weight_mask

        lin = nn.Linear(dim_in, dim_out)
        self._weights = lin.weight
        self._bias = lin.bias

        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w = torch.mul(self._weight_mask, self._weights)
        res = torch.addmm(self._bias, x, w.transpose(0, 1))

        hyper_bias = self._hyper_bias(t.view(1, 1))
        return res * torch.sigmoid(self._hyper_gate(t.view(1, 1))) + hyper_bias


class WeilbachSparseODENet(nn.Module):
    """Implements the sparse ODENet described in:
    http://proceedings.mlr.press/v108/weilbach20a/weilbach20a.pdf

    Code taken from: https://github.com/plai-group/daphne with minor changes
    to adapt to PyTorch Lightning.

    AS the linear layers enforce independencies by directly multiplying the
    adjacency matrix and offers no factorization, multiple consecutive hidden
    layers are not possible, nor are hidden layers that are not the size of
    the adjacency matrix.
    """
    def __init__(self,
                 input_dim: int,
                 adj_mat: torch.Tensor,
                 num_layer: int = 4,
                 act_type: str = 'tanh'):
        super(WeilbachSparseODENet, self).__init__()

        layers = [WeilbachSparseLinear(input_dim+1, input_dim, adj_mat)]

        for i in range(num_layer-1):
            layers.append(WeilbachSparseLinear(input_dim, input_dim, adj_mat))

        self.layers = nn.ModuleList(layers)

        activation_fn = NONLINEARITIES[act_type]
        activation_fns = [activation_fn for _ in range(num_layer)]
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_dim = x.shape[0]
        dx = torch.cat([x, t * torch.ones([batch_dim, 1])], dim=1)

        for i, layer in enumerate(self.layers):
            # if not last layer, use nonlinearity
            if i < len(self.layers) - 1:
                acti = layer(t, dx)
                if i == 0:
                    dx = self.activation_fns[i](acti)
                else:
                    dx = self.activation_fns[i](acti) + dx
            else:
                dx = layer(t, dx)
        return dx


class FCODEnet(nn.Module):
    """Fully connected ODENet.

    Implementation taken from FFJORD: https://github.com/rtqichen/ffjord, in
    particular from the IgnoreLinear class.

    Although better FFJORD ODENets exists, this fully connection layer is the
    best negative control for against the StrODENet and WeilbachSparseODENet.
    """
    def __init__(self,
                 input_shape: list[int],
                 hidden_dims: list[int],
                 act_type: str):
        super(FCODEnet, self).__init__()

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out in hidden_dims + (input_shape[0],):
            layers.append(nn.Linear(hidden_shape[0], dim_out))
            activation_fns.append(NONLINEARITIES[act_type])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        for i, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if i < len(self.layers) - 1:
                dx = self.activation_fns[i](dx)
        return dx


class StrODENet(StrNN):
    """Intializes a StrNN model an ODE dynamics function."""

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Wraps superclass forward to take time variable."""
        return super().forward(x)
