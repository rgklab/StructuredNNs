import numpy as np

import torch

from strnn import StrNN
from .conditioner import Conditioner


class StrNNConditioner(Conditioner):
    """Normalizing Flow Conditioner using StrNN to compute flow parameters."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int],
        n_out_param: int,
        act_type: str,
        adj_mat: np.ndarray,
        opt_type: str,
        opt_args: dict
    ):
        """Initialize an StrNN conditioner.

        Args:
            input_dim: Dimension of input.
            hidden_dim: List of hidden widths for each layer.
            n_out_param: Number of output parameters per input variable.
            act_type: Activation function used in StrNN.
            adj_mat: 2D binary adjacency matrix.
            out_type: Matrix factorization used by StrNN.
            opt_args: Dictionary of arguments to pass to factorization.
        """
        super().__init__()
        self.input_dim = input_dim
        self.strnn = StrNN(input_dim, hidden_dim, input_dim * n_out_param,
                           opt_type, opt_args, None, adj_mat, act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass for StrNN Conditioner.

        Output is reshaped so all output that corresponds to a variable is
        grouped in the last dimension. Care must be taken so that outputs which
        do not respect the independence relations for a variable are grouped.

        The StrNN stacks blocks of outputs, e.g., [1 2 3 4 1 2 3 4] instead of
        [1 1 2 2 3 3 4 4], and the reshape operation must follow this.
        Args:
            x: Input data
        Returns:
            Parameters for use in Normalizer.
        """
        return self.strnn(x).view(x.shape[0], -1, x.shape[1]).permute(0, 2, 1)
