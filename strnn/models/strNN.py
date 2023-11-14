import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from strnn.models.model_utils import NONLINEARITIES


class MaskedLinear(nn.Linear):
    """
    A linear neural network layer, except with a configurable
    binary mask on the weights
    """
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        # register_buffer used for non-parameter variables in the model
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask: np.ndarray):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

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
        activation: str = 'relu'
    ):
        """
        Initializes a Structured Neural Network (StrNN)
        :param nin: input dimension
        :param hidden_sizes: list of hidden layer sizes
        :param nout: output dimension
        :param opt_type: optimization type: greedy, zuko, MADE
        :param opt_args: additional optimization algorithm params
        :param precomputed_masks: previously stored masks, use directly
        :param adjacency: the adjacency matrix, nout by nin
        :param activation: activation function to use in this NN
        """
        super().__init__()

        # Set parameters
        self.nin = nin
        self.hidden_sizes = hidden_sizes
        self.nout = nout
        self.opt_type = opt_type
        self.opt_args = opt_args
        self.precomputed_masks = precomputed_masks
        self.A = adjacency

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
        else:
            # All per-layer factorization algos
            for l in self.hidden_sizes:
                if self.opt_type == 'greedy':
                    (M1, M2) = self.factorize_single_mask_greedy(adj_mtx, l)
                elif self.opt_type == 'MADE':
                    (M1, M2) = self.factorize_single_mask_MADE(adj_mtx, l)
                else:
                    raise ValueError(f'{self.opt_type} is NOT an implemented optimization type!')

                # Update the adjacency structure for recursive call
                adj_mtx = M1
                masks = masks + [M2.T]  # take transpose for size: (n_inputs x n_hidden/n_output)
            masks = masks + [M1.T]

        return masks

    def factorize_single_mask_greedy(
        self,
        adj_mtx: np.ndarray,
        n_hidden: int
    ) -> (np.ndarray, np.ndarray):
        """
        Factorize adj_mtx into M1 * M2
        :param adj_mtx: adjacency structure, n_outputs x n_inputs
        :param n_hidden: number of units in this hidden layer
        :return: masks:
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

    def factorize_single_mask_MADE(self, adj_mtx: np.ndarray, n_hidden: int):
        return None, None

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
        """
        Given the model's masks, [M1, M2, ..., Mk],
        check if the matrix product
        (M1 * M2 * ... * Mk).T = Mk.T * ... * M2.T * M1.T
        respects A's adjacency structure.
        :return: True or False
        """
        mask_prod = self.masks[0]
        for i in range(1, len(self.masks)):
            mask_prod = mask_prod @ self.masks[i]
        mask_prod = mask_prod.T

        constraint = (mask_prod > 0.0001) * 1. - self.A
        if np.any(constraint != 0.):
            return False
        else:
            return True


if __name__ == '__main__':
    # TODO: Write unit tests
    # Test StrNN model
    d = 5
    A = np.ones((d, d))
    A = np.tril(A, -1)

    model = StrNN(
        nin=d,
        hidden_sizes=(d * 2,),
        nout=d,
        opt_type='greedy',
        opt_args={'var_penalty_weight': 0.0},
        precompute_masks=None,
        adjacency=None,
        activation='relu')
