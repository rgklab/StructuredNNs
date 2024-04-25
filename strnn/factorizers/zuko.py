import numpy as np
import torch


class ZukoFactorizer:
    """Implements factorization algorithm from Zuko repository."""

    def __init__(self, adjacency: np.ndarray, opt_args: dict | None = None):
        """Initialize Zuko Factorizer.

        Args:
            adjacency: Global adjacency matrix to factorize.
            opt_args: Unused.
        """
        self.adjacency = adjacency
        self.opt_args = opt_args

    def factorize(self, hidden_sizes: tuple[int, ...]) -> list[np.ndarray]:
        """Factorize adjacency matrix according to Zuko algorithm.

        Implementation adapted from: https://github.com/probabilists/zuko.

        Args:
            hidden_sizes: List of hidden widths of intermediate layers.

        Returns:
            List of all weight masks.
        """
        masks = []

        adj_mtx = torch.from_numpy(self.adjacency)
        n_outputs = self.adjacency.shape[0]

        A_prime, inv = torch.unique(adj_mtx, dim=0, return_inverse=True)
        n_deps = A_prime.sum(dim=-1)
        P = (A_prime @ A_prime.T == n_deps).double()

        indices = None

        for i, h_i in enumerate((*hidden_sizes, n_outputs)):
            if i > 0:
                # Not the first mask
                mask = P[:, indices]
            else:
                # First mask: just use rows from A
                mask = A_prime

            if sum(sum(mask)) == 0.0:
                raise ValueError("Adjacency matrix will yield null Jacobian.")

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
