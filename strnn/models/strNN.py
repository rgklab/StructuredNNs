import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from strnn.factorizers import check_masks, GreedyFactorizer, \
    GreedyParallelFactorizer, MADEFactorizer, ZukoFactorizer
from strnn.models.model_utils import NONLINEARITIES
from adpativeLayerNorm import AdaptiveLayerNorm


OPT_MAP = {
    "greedy": GreedyFactorizer,
    "greedy_parallel": GreedyParallelFactorizer,
    "made": MADEFactorizer,
    "zuko": ZukoFactorizer,
}


def ian_initialization(
    weights: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    a: float = 0,
    nonlinearity: str = 'leaky_relu',
    distribution: str = 'normal'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sparsity-aware weight initialization scheme for masked NNs

    Fills the input weights with values based on Kaiming normal initialization
    (https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_normal_)
    but takes into account the number of fan-ins that are masked in StrNN

    Set distribution = 'uniform' for uniform initialization

    :param weights: weight tensor to be filled
    :param bias: bias tensor to be filled
    :param mask: weight mask for this layer
    :param a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
    :param nonlinearity: the non-linear function (`nn.functional` name), recommended to use only 'relu' or 'leaky_relu'
    :param 
    """
    # Ensuring non-zero fan-in for each unit
    fan_ins = mask.sum(dim=1).clamp(min=1)

    # Calculate the appropriate gain for the nonlinearity
    gain = torch.nn.init.calculate_gain(nonlinearity, a)

    # Compute standard deviation
    std = gain / torch.sqrt(fan_ins)

    if distribution == 'uniform':
        # Compute bounds for uniform distribution
        bounds = std * math.sqrt(3.0)

    if distribution == 'normal':
        # Apply normal initialization row-wise to handle varying fan-in
        with torch.no_grad():
            for i, std_dev in enumerate(std):
                weights[i].normal_(0, std_dev.item())
    elif distribution == 'uniform':
        # Compute bounds for uniform distribution
        bounds = std * math.sqrt(3.0)

        # Apply uniform initialization row-wise to handle varying fan-in
        with torch.no_grad():
            for i, bound in enumerate(bounds):
                weights[i].uniform_(-bound.item(), bound.item())
    else:
        raise ValueError(f"Invalid distribution for ian init: {distribution}")

    return weights, bias


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
            init: str,
            activation: str = "relu"
    ):
        """Initialize MaskedLinear layer.

        Args:
            in_features: Feature dimension of input data.
            out_features: Feature dimension of output.
            activation: Unused.
        """
        super().__init__(in_features, out_features)
        # register_buffer used for non-parameter variables in the model
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self.init = init
        self.activation = activation
        

    def reset_parameters_w_masking(self) -> None:
        """
        Reset parameters in the network based on the initialization scheme

        Setting a=sqrt(5) in kaiming_uniform (thus also ian_uniform) is the
        same as initializing with uniform(-1/sqrt(in_features), 1/sqrt(in_features)).
        For details, see https://github.com/pytorch/pytorch/issues/57109
        """
        if self.init == 'ian_uniform':
            ian_initialization(
                self.weight, self.bias, self.mask, a=math.sqrt(5), 
                nonlinearity=self.activation, distribution='uniform'
            )
        elif self.init == 'ian_normal':
             ian_initialization(
                self.weight, self.bias, self.mask, a=math.sqrt(5), 
                nonlinearity=self.activation, distribution='normal'
            )
        elif self.init == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(
                self.weight, a=math.sqrt(5),
                nonlinearity=self.activation
            )
        elif self.init == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(
                self.weight, a=math.sqrt(5), 
                nonlinearity=self.activation
            )

        with torch.no_grad():
            self.bias.zero_()


    def set_mask(self, mask: np.ndarray):
        """Store mask for use during forward pass, then re-initialize weights
        based on initialization scheme.

        Arg:
            mask: Mask applied to weight matrix.
        """
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        self.reset_parameters_w_masking()


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
        opt_type: str = "greedy",
        opt_args: dict = {"var_penalty_weight": 0.0},
        precomputed_masks: np.ndarray | None = None,
        adjacency: np.ndarray | None = None,
        activation: str = "relu",
        init_type: str = 'ian_uniform',
        norm_type: str | None = None
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
        
        # Set up initialization and normalization schemes
        self.init_type = init_type
        self.norm_type = norm_type

        # Define StrNN network
        self.net_list = []
        hs = [nin] + list(hidden_sizes) + [nout]  # list of all layer sizes

        # Create MaskedLinear and normalizations for each hidden layer
        for h0, h1 in zip(hs[:-1], hs[1:-1]):
            self.net_list.append(MaskedLinear(h0, h1, self.init_type, activation))

            # Add normalization layer
            if norm_type == 'layer':
                self.net_list.append(nn.LayerNorm(h1))
            elif norm_type == 'batch':
                self.net_list.append(nn.BatchNorm1d(h1))
            elif norm_type == 'adaptive_layer':
                pass
            else:
                if norm_type is not None:
                    raise ValueError(f"Invalid normalization type: {norm_type}")
        
            # Add the activation function
            self.net_list.append(NONLINEARITIES[activation])

        # Last layer: no normalization or activation
        self.net_list.append(MaskedLinear(hs[-2], hs[-1], self.init_type, activation))

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
                                  " factorizer is not MADE."))

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
        mask_idx = 0
        for layer in self.net:
            if isinstance(layer, MaskedLinear):
                layer.set_mask(self.masks[mask_idx])
                mask_idx += 1


if __name__ == '__main__':
    adj_mtx = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ])

    model = StrNN(
        nin=4,
        hidden_sizes=[6, 8, 6],
        nout=4,
        opt_type="greedy",
        adjacency=adj_mtx,
        init_type='ian_uniform'
    )

    print(model.masks)


