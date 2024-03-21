import numpy as np

import torch
import torch.nn as nn

from strnn.models. model_utils import NONLINEARITIES
from .conditioner import Conditioner


class GNFMLP(nn.Module):
    """MLP used in GNF conditioner.

    Code taken from: https://github.com/AWehenkel/Graphical-Normalizing-Flows.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int],
        output_dim: int,
        act_type: str
    ):
        """Initialize a GNF MLP.

        Args:
            input_dim: Input dimension.
            hidden_dim: List of hidden widths.
            output_dim: Output dimension.
            act_type: Activation function used between layers.
        """
        super().__init__()

        act_fn = NONLINEARITIES[act_type]

        l1 = [input_dim] + list(hidden_dim)
        l2 = list(hidden_dim) + [output_dim]

        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), act_fn]
        layers.pop()

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute GNFMLP forward pass."""
        return self.net(x)


class GNFConditioner(Conditioner):
    """Conditioner from Graphical Normalizing Flows.

    Code taken from: https://github.com/AWehenkel/Graphical-Normalizing-Flows.

    Uses input masking to allow neural network outputs to respect a prescribed
    adjacency. Original code contains methods to allow optimization of
    adjacency matrix based on data. While helpful, this was unrelated to our
    method so the code was removed for clarity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int],
        n_out_param: int,
        act_type: str,
        hot_encoding: bool,
        adj_mat: np.ndarray
    ):
        """Initialize a GNF Conditioner.

        Args:
            input_dim: Input dimension.
            hidden_dim: Tuple of hidden widths.
            n_out_param: Number of output parameters per input variable.
            act_type: Activation function used between layers.
        """
        super().__init__()
        self.adj_mat = torch.Tensor(adj_mat)

        self.input_dim = input_dim
        self.hot_encoding = hot_encoding

        in_net = input_dim * 2 if hot_encoding else input_dim

        self.embedding_net = GNFMLP(in_net, hidden_dim, n_out_param, act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of GNF Conditioner.

        Note that each GNF forward pass is equivalent to calling the underlying
        GNFMLP D times.

        Args:
            x: Input data. Has shape (Batch Size x Feature Dimension).

        Returns:
            Conditioner output.
        """
        # Expand adjacency to batch size
        mask = self.adj_mat.unsqueeze(0).expand(x.shape[0], -1, -1).to(x)

        # Duplicate input by number of variables. Each variable requires its
        # own mask.
        batch = x.unsqueeze(1).expand(-1, self.input_dim, -1)

        masked_input = (batch * mask).reshape(x.shape[0] * self.input_dim, -1)

        if self.hot_encoding:
            # Append one hot encoding to inputs
            encoding = torch.eye(self.input_dim).to(x)
            encoding = encoding.unsqueeze(0).expand(x.shape[0], -1, -1)
            encoding = encoding.contiguous().view(-1, self.input_dim)

            masked_input = torch.cat((masked_input, encoding), 1)

        out = self.embedding_net(masked_input)
        out = out.view(x.shape[0], self.input_dim, -1)

        return out
