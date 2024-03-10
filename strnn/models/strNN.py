import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import math

from strnn.models.model_utils import NONLINEARITIES


def ian_uniform(
        weights: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        a: float = 0,
        nonlinearity: str = 'leaky_relu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fills the input weights with values based on Kaiming uniform initialization
    (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_)
    but takes into account the number of fan_ins that are masked in StrNN

    :param tensor: weight tensor to be filled
    :param mask: weight mask for this layer
    :param
    """
    if torch.overrides.has_torch_function_variadic(weights):
        return torch.overrides.handle_torch_function(
            ian_uniform,
            (weights,),
            tensor=weights,
            a=a,
            nonlinearity=nonlinearity)

    if 0 in weights.shape or 0 in bias.shape:
        warnings.warn("In ian_uniform: weights or bias cannot be 0-dimensional!")
        return weights, bias

    # Compute fan_in based on mask
    fan_ins = mask.sum(dim=1)

    # Compute other quantities as in Kaiming uniform
    gain = torch.nn.init.calculate_gain(nonlinearity, a)

    # Update weights
    i = 0
    for row in weights:
        fan_in = fan_ins[i]
        std = gain / math.sqrt(fan_in) if fan_in > 0 else gain
        bound = math.sqrt(3.0) * std

        with torch.no_grad():
            row.uniform_(-bound, bound)
        i += 1

    # Update bias similarly
    i = 0
    for b in bias:
        fan_in = fan_ins[i]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        with torch.no_grad():
            b.uniform_(-bound, bound)
        i += 1

    return weights, bias


class MaskedLinear(nn.Linear):
    """
    A linear neural network layer, except with a configurable
    binary mask on the weights
    """
    mask: torch.Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            ian_init: bool = False,
            activation: str = 'relu'
    ):
        super().__init__(in_features, out_features)
        # register_buffer used for non-parameter variables in the model
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.ian_init = ian_init
        self.activation = activation

    def reset_parameters_w_masking(self) -> None:
        """
        Setting a=sqrt(5) in kaiming_uniform (thus also ian_uniform) is the
        same as initializing with uniform(-1/sqrt(in_features), 1/sqrt(in_features)).
        For details, see https://github.com/pytorch/pytorch/issues/57109
        """
        ian_uniform(
            self.weight, self.bias, self.mask,
            a=math.sqrt(5), nonlinearity=self.activation
        )

    def set_mask(self, mask: np.ndarray):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        if self.ian_init:
            # Reinitialize weights based on masks
            self.reset_parameters_w_masking()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # * is element-wise multiplication in numpy
        return F.linear(input, self.mask * self.weight, self.bias)


class StrNN(nn.Module):
    """
    Main neural network class that implements a Structured Neural Network
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
        activation: str = 'relu',
        ian_init: bool = False
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
            ian_init: weight initialization takes the masks into account
        """
        super().__init__()

        # Set parameters
        self.nin = nin
        self.hidden_sizes = hidden_sizes
        self.nout = nout
        self.opt_type = opt_type
        self.opt_args = opt_args
        self.precomputed_masks = precomputed_masks

        if adjacency is not None:
            self.A = adjacency
        else:
            assert self.opt_type == 'MADE'
            # Initialize adjacency structure to fully autoregressive
            self.A = np.tril(np.ones((nout, nin)), -1)

        self.ian_init = ian_init

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
                MaskedLinear(h0, h1, self.ian_init),
                self.activation
            ])

        # Remove the last activation for the output layer
        self.net_list.pop()
        self.net = nn.Sequential(*self.net_list)

        # Update masks
        self.update_masks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagates the input forward through the StrNN network
        :param x: input, sample_size by data_dimensions
        :return: output, sample_size by output_dimensions
        """
        return self.net(x)

    def update_masks(self):
        """
        :return: None
        """
        if self.precomputed_masks is not None:
            # Load precomputed masks if provided
            masks = self.precomputed_masks
        else:
            masks = self.factorize_masks()

        self.masks = masks
        assert self.check_masks(), "Mask check failed!"

        # For when each input produces multiple outputs
        # e.g.: each x_i gives mean and variance for Gaussian density estimation
        if self.nout != self.A.shape[0]:
            # Then nout should be an exact multiple of nin
            assert self.nout % self.nin == 0
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
        # Set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for layer, mask in zip(layers, self.masks):
            layer.set_mask(mask)

    def factorize_masks(self) -> list[np.ndarray]:
        """
        Factorize the given adjacency structure into per-layer masks.
        We use a recursive approach here for efficiency and simplicity.
        :return: list of masks in order for layers from inputs to outputs.
        This order matches how the masks are assigned to the networks in MADE.
        """
        masks: list[np.ndarray] = []
        adj_mtx = np.copy(self.A)

        # TODO: We need to handle default None case (maybe init an FC layer?)

        if self.opt_type == 'Zuko':
            # Zuko creates all masks at once
            masks = self.factorize_masks_zuko(self.hidden_sizes)
        elif self.opt_type == 'MADE':
            masks = self.factorize_masks_MADE()
        else:
            # All per-layer factorization algos
            for lyr in self.hidden_sizes:
                if self.opt_type == 'greedy':
                    (M1, M2) = self.factorize_single_mask_greedy(adj_mtx, lyr)
                else:
                    err_msg = "{} is NOT an implemented optimization type!"
                    raise ValueError(err_msg.format(self.opt_type))

                # Update the adjacency structure for recursive call
                adj_mtx = M1
                # Take transpose for size: (n_inputs x n_hidden/n_output)
                masks = masks + [M2.T]
            masks = masks + [M1.T]

        return masks

    def factorize_single_mask_greedy(
        self,
        adj_mtx: np.ndarray,
        n_hidden: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Factorize adj_mtx into M1 * M2.

        Args:
            adj_mtx: adjacency structure, n_outputs x n_inputs
            n_hidden: number of units in this hidden layer

        Returns:
            Masks (M1, M2) with the shapes:
                M1 size: (n_outputs x n_hidden)
                M2 size: (n_hidden x n_inputs)
        """
        # find non-zero rows and define M2
        A_nonzero = adj_mtx[~np.all(adj_mtx == 0, axis=1), :]
        n_nonzero_rows = A_nonzero.shape[0]
        M2 = np.zeros((n_hidden, adj_mtx.shape[1]))
        for i in range(n_hidden):
            M2[i, :] = A_nonzero[i % n_nonzero_rows]

        # find each row of M1
        M1 = np.ones((adj_mtx.shape[0], n_hidden))
        for i in range(M1.shape[0]):
            # Find indices where A is zero on the ith row
            Ai_zero = np.where(adj_mtx[i, :] == 0)[0]
            # find row using closed-form solution
            # find unique entries (rows) in j-th columns of M2 where Aij = 0
            row_idx = np.unique(np.where(M2[:, Ai_zero] == 1)[0])
            M1[i, row_idx] = 0.0

        return M1, M2

    def factorize_masks_MADE(self) -> list[np.ndarray]:
        """Factorize adjacency matrix according to MADE algorithm.

        Non-random version of the MADE factorization algorithm is used.

        Returns:
            List of all weight masks.
        """
        self.m = {}
        n_layers = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin)
        for layer in range(n_layers):
            self.m[layer] = np.array([self.nin - 1 - (i % self.nin) for i in range(self.hidden_sizes[layer])])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(n_layers)]
        masks.append(self.m[n_layers - 1][:, None] < self.m[-1][None, :])

        return masks

    def factorize_masks_zuko(
        self,
        hidden_sizes: tuple[int]
    ) -> list[np.ndarray]:
        masks = []

        adj_mtx = torch.from_numpy(self.A)
        A_prime, inv = torch.unique(adj_mtx, dim=0, return_inverse=True)
        n_deps = A_prime.sum(dim=-1)
        P = (A_prime @ A_prime.T == n_deps).double()

        for i, h_i in enumerate((*hidden_sizes, self.nout)):
            if i > 0:
                # Not the first mask
                mask = P[:, indices]
            else:
                # First mask: just use rows from A
                mask = A_prime

            if sum(sum(mask)) == 0.0:
                raise ValueError("The adjacency matrix leads to a null Jacobian.")

            if i < len(hidden_sizes):
                # Still on intermediate masks
                reachable = mask.sum(dim=-1).nonzero().squeeze(dim=-1)
                indices = reachable[torch.arange(h_i) % len(reachable)]
                mask = mask[indices]
            else:
                # We are at the last mask
                mask = mask[inv]

            # Need to transpose all masks to match other algorithms
            masks.append(mask.T.numpy())

        return masks

    def check_masks(self) -> bool:
        """Check whether weight masks respect prescribed adjacency matrix.

        Given the model's masks, [M1, M2, ..., Mk], check if the matrix product
        (M1 * M2 * ... * Mk).T = Mk.T * ... * M2.T * M1.T
        respects A's adjacency structure.

        Returns:
            True or False depending on validity of weight masks.
        """
        mask_prod = self.masks[0]
        for i in range(1, len(self.masks)):
            mask_prod = mask_prod @ self.masks[i]
        mask_prod = mask_prod.T

        constraint = (mask_prod > 0.0001) * 1. - self.A

        return not np.any(constraint != 0.)
