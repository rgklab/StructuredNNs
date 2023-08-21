import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from .models.model_utils import NONLINEARITIES


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


class StrNN(pl.LightningModule):
    """
    Main neural network class that implements a Structured Neural Network
    Can also become a MADE or Zuko masked NN by specifying the opt_type flag
    """
    def __init__(
            self,
            nin: int, hidden_sizes: tuple[int, ...], nout: int,
            opt_type: str = 'greedy',
            opt_args: dict = {'var_penalty_weight': 0.0},
            precompute_masks: np.ndarray | None = None,
            adjacency: np.ndarray | None = None,
            activation: str = 'relu'):
        """
        Initializes a Structured Neural Network (StrNN)
        :param nin:
        :param hidden_sizes:
        :param nout:
        :param opt_type:
        :param opt_args:
        :param precompute_masks:
        :param adjacency:
        :param activation:
        """
        super().__init__()
        self.save_hyperparameters()

        # Set parameters
        self.nin = nin
        self.hidden_sizes = hidden_sizes
        self.nout = nout
        self.opt_type = opt_type
        self.opt_args = opt_args
        self.precompute_masks = precompute_masks
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
        if self.precompute_masks is not None:
            # Load precomputed masks if provided
            masks = self.precompute_masks
        else:
            masks = self.factorize_masks()

        self.masks = masks
        self.check_masks()

        # Handle the case where nout = nin * k, for integer k > 1
        # Set the last mask to have k times the number of outputs
        if self.nout > self.nin:
            assert self.nout % self.nin == 0
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            self.masks[-1] = np.concatenate([self.masks[-1]] * k, axis=1)

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
            raise NotImplementedError
            masks = self.factorize_masks_zuko(adj_mtx)

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

    def factorize_single_mask_greedy(self, adj_mtx: np.ndarray, n_hidden: int):
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

    def check_masks(self) -> bool:
        """
        Given the model's masks, [M1.T, M2.T, ..., Mk.T],
        check if the matrix product Mk*...*M2*M1 respects A's
        adjacency structure.
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
        hidden_sizes=(d*2,),
        nout=d,
        opt_type='greedy',
        opt_args={'var_penalty_weight': 0.0},
        precompute_masks=None,
        adjacency=None,
        activation='relu')
