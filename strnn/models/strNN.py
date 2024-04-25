import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from strnn.factorizers import check_masks
from strnn.factorizers import GreedyFactorizer, MADEFactorizer, ZukoFactorizer
from strnn.models.model_utils import NONLINEARITIES


OPT_MAP = {
    "greedy": GreedyFactorizer,
    "made": MADEFactorizer,
    "zuko": ZukoFactorizer,
}


class MaskedLinear(nn.Linear):
    """Weight-masked lienar layer.

    A linear neural network layer, except with a configurable binary mask
    on the weights.
    """

    mask: torch.Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: str = 'relu'
    ):
        """Initialize MaskedLinear layer.

        Args:
            in_features: Feature dimension of input data.
            out_features: Feature dimension of output.
            activation: Unused.
        """
        super().__init__(in_features, out_features)
        # register_buffer used for non-parameter variables in the model
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.activation = activation

    def set_mask(self, mask: np.ndarray):
        """Store mask for use during forward pass.

        Arg:
            mask: Mask applied to weight matrix.
        """
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute forward pass for MaskedLinear.

        Applies weight mask to weight matrix to block desired connections.

        Args:
            input: Input data.

        Returns:
            MaskedLinear output.
        """
        # Note: * is element-wise multiplication in numpy
        return F.linear(input, self.mask * self.weight, self.bias)


class StrNN(nn.Module):
    """Main neural network class that implements a Structured Neural Network.

    Can also become a MADE or Zuko masked NN by specifying the opt_type flag
    """

    def __init__(
        self,
        nin: int,
        hidden_sizes: tuple[int, ...],
        nout: int,
        opt_type: str = 'greedy',
        opt_args: dict = {'var_penalty_weight': 0.0},
        precomputed_masks: np.ndarray | None = None,
        adjacency: np.ndarray | None = None,
        activation: str = 'relu'
    ):
        """Initialize a Structured Neural Network (StrNN).

        Args:
            nin: input dimension
            hidden_sizes: list of hidden layer sizes
            nout: output dimension
            opt_type: optimization type: greedy, zuko, MADE
            opt_args: additional optimization algorithm params
            precomputed_masks: previously stored masks, use directly
            adjacency: the adjacency matrix, nout by nin
            activation: activation function to use in this NN
        """
        super().__init__()

        # Set parameters
        self.nin = nin
        self.hidden_sizes = hidden_sizes
        self.nout = nout

        # Define activation
        try:
            self.activation = NONLINEARITIES[activation]
        except ValueError:
            raise ValueError(f"{activation} is not a valid activation!")

        # Define StrNN network
        self.net_list = []
        hs = [nin] + list(hidden_sizes) + [nout]  # list of all layer sizes
        for h0, h1 in zip(hs, hs[1:]):
            self.net_list.extend([
                MaskedLinear(h0, h1),
                self.activation
            ])

        # Remove the last activation for the output layer
        self.net_list.pop()
        self.net = nn.Sequential(*self.net_list)

        # Load adjacency matrix
        self.opt_type = opt_type.lower()
        self.opt_args = opt_args

        if adjacency is not None:
            self.A = adjacency
        else:
            if self.opt_type == "made":
                # Initialize adjacency structure to fully autoregressive
                warnings.warn(("Adjacency matrix is unspecified, defaulting to"
                               " fully autoregressive structure."))
                self.A = np.tril(np.ones((nout, nin)), -1)
            else:
                raise ValueError(("Adjacency matrix must be specified if"
                                  "factorizer is not MADE."))

        # Setup adjacency factorizer
        try:
            self.factorizer = OPT_MAP[self.opt_type](self.A, self.opt_args)
        except ValueError:
            raise ValueError(f"{opt_type} is not a valid opt_type!")

        self.precomputed_masks = precomputed_masks

        # Update masks
        self.update_masks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input forward through the StrNN network.

        Args:
            x: Input of size (sample_size by data_dimensions)
        Returns:
            Output of size (sample_size by output_dimensions)
        """
        return self.net(x)

    def update_masks(self):
        """Update masked linear layer masks to respect adjacency matrix."""
        if self.precomputed_masks is not None:
            # Load precomputed masks if provided
            masks = self.precomputed_masks
        else:
            masks = self.factorizer.factorize(self.hidden_sizes)

        self.masks = masks
        assert check_masks(masks, self.A), "Mask check failed!"

        # For when each input produces multiple outputs
        # e.g. each x_i gives mean and variance for Gaussian density estimation
        if self.nout != self.A.shape[0]:
            # Then nout should be an exact multiple of nin
            assert self.nout % self.nin == 0
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # Set the masks in all MaskedLinear layers
        layers = [m for m in self.net.modules() if isinstance(m, MaskedLinear)]
        for layer, mask in zip(layers, self.masks):
            layer.set_mask(mask)
