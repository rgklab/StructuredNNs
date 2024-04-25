from abc import ABCMeta, abstractmethod

import numpy as np

import torch
import torch.nn as nn

from strnn.models.strNN import StrNN
from ..model_utils import NONLINEARITIES

from ...models import Array_like


class ODENet(nn.Module, metaclass=ABCMeta):
    """Interface for networks used to model ODE dynamics."""

    @abstractmethod
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Abstract method for ODENet forward pass."""
        pass


class WeilbachSparseLinear(nn.Module):
    """Concatenated squash linear layer.

    See: http://proceedings.mlr.press/v108/weilbach20a/weilbach20a.pdf.
    Implementation adapted from https://github.com/plai-group/daphne.
    """

    _weight_mask: torch.Tensor

    def __init__(self, dim_in: int, dim_out: int, adj_mat: Array_like):
        """Initialize a sparse concat squash layer as in Weilbach et al. 2020.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            adj_mat: 2D binary adjacency matrix.
        """
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.adj_mat = torch.Tensor(adj_mat)

        self.register_buffer("_weight_mask", torch.zeros([dim_out, dim_in]))
        self._weight_mask[:adj_mat.shape[0], :adj_mat.shape[1]] = self.adj_mat

        lin = nn.Linear(dim_in, dim_out)
        self._weights = lin.weight
        self._bias = lin.bias

        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            t: Time of integration.
            x: Data input.

        Returns:
            Predicted dynamics.
        """
        w = torch.mul(self._weight_mask, self._weights)
        res = torch.addmm(self._bias, x, w.transpose(0, 1))

        hyper_bias = self._hyper_bias(t.view(1, 1))
        return res * torch.sigmoid(self._hyper_gate(t.view(1, 1))) + hyper_bias


class WeilbachSparseODENet(ODENet):
    """Implements the sparse ODENet described below.

    http://proceedings.mlr.press/v108/weilbach20a/weilbach20a.pdf

    Code adapted from: https://github.com/plai-group/daphne.

    As linear layers enforce independencies by directly multiplying the
    adjacency matrix and offers no factorization, multiple consecutive hidden
    layers are not possible, nor are hidden layers that are not the size of
    the adjacency matrix.
    """

    def __init__(
        self,
        input_dim: int,
        num_layer: int,
        act_type: str,
        adj_mat: Array_like
    ):
        """Initialize sparse ODENet from Weilbach et al. 2020.

        Args:
            input_dim: Input dimension.
            num_layer: Number of stacked WeilbachSparseLinear layers.
            act_type: Type of activation function used between layers.
            adj_mat: 2D Binary adjacency matrix.
        """
        super().__init__()

        adj_mat = torch.Tensor(adj_mat)
        self.adj_mat = adj_mat

        layers = [WeilbachSparseLinear(input_dim + 1, input_dim, adj_mat)]

        for _ in range(num_layer):
            layers.append(WeilbachSparseLinear(input_dim, input_dim, adj_mat))

        self.layers = nn.ModuleList(layers)

        activation_fn = NONLINEARITIES[act_type]
        activation_fns = [activation_fn for _ in range(num_layer)]
        self.activation_fns = nn.ModuleList(activation_fns)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            t: Time of integration.
            x: Data input.

        Returns:
            Predicted dynamics.
        """
        batch_dim = x.shape[0]
        dx = torch.cat([x, t * torch.ones([batch_dim, 1]).to(x)], dim=1)

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


class IgnoreLinear(nn.Module):
    """Fully connected layer for use in ODE Net.

    Code copied from FFJORD: https://github.com/rtqichen/ffjord.
    """

    def __init__(self, dim_in: int, dim_out: int):
        """Initialize IgnoreLinear layer.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
        """
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass using dummy layer.

        Args:
            t: Time of integration, unused.
            x: Data values.

        Returns:
            Predicted dynamics.
        """
        return self._layer(x)


class FCODEnet(ODENet):
    """Fully connected ODENet.

    Code modified from FFJORD: https://github.com/rtqichen/ffjord.

    Although better FFJORD ODENets exists, this fully connection layer is the
    best negative control for against the StrODENet and WeilbachSparseODENet.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int],
        act_type: str
    ):
        """Initialize a fully connected ODENet.

        Args:
            input_shape: Input dimension.
            hidden_dims: List containing widths of hidden dimensions.
            act_type: Activation function between layers.

        Returns:
            Fully connected NN for use in Neural ODE.
        """
        super().__init__()

        layers = []
        activation_fns = []
        hidden_shape = input_dim

        for dim_out in list(hidden_dims) + [input_dim]:
            layers.append(IgnoreLinear(hidden_shape, dim_out))
            activation_fns.append(NONLINEARITIES[act_type])

            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            t: Time of integration.
            x: Data input.

        Returns:
            Predicted dynamics.
        """
        dx = y
        for i, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if i < len(self.layers) - 1:
                dx = self.activation_fns[i](dx)
        return dx


class StrODENet(ODENet):
    """Wraps StrNN model for use as an ODE dynamic function."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int],
        activation: str,
        opt_type: str,
        opt_args: dict,
        adjacency: np.ndarray
    ):
        """Initialize StrODENet.

        Args:
            input_dim: Input dimension.
            hidden_dim: List of StrNN hidden widths per layer.
            activation: Activation function between layers.
            opt_type: Factorizer used by StrNN.
            opt_args: Arguments to pass to factorizer.
            adjacency: Adjacency matrix.
        """
        super().__init__()
        self.net = StrNN(
            input_dim,
            hidden_dim,
            input_dim,
            opt_type,
            opt_args,
            None,
            adjacency,
            activation
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute dynamics using StrNN.

        Currently uses an autonomous function.

        Args:
            t: Time of integration, unused.
            x: Data values.

        Returns:
            Predicted dynamics.
        """
        return self.net(x)
