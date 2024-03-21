import numpy as np

from .factorizer import AdjacencyFactorizer


class MADEFactorizer(AdjacencyFactorizer):
    """Implements factorization algorithm from MADE."""

    def __init__(self, adjacency: np.ndarray, opt_args: dict | None = None):
        """Initialize MADE factorizer.

        Factorizer and only be applied on fully autoregressive adjacencies.

        Args:
            adjacency: Unused, but must be fully lower triangular.
            opt_args: Unused.
        """
        if not np.all(adjacency == np.tril(np.ones_like(adjacency), -1)):
            raise ValueError(("MADE can only be used on a fully autoregressive"
                              " adjacency matrix."))
        self.adjacency = adjacency
        self.opt_args = opt_args

    def factorize(self, hidden_sizes: tuple[int, ...]) -> list[np.ndarray]:
        """Factorize adjacency matrix according to MADE algorithm.

        Non-random version of the MADE factorization algorithm is used.

        Args:
            hidden_sizes: List of hidden widths of intermediate layers.

        Returns:
            List of all weight masks.
        """
        m = {}
        n_layers = len(hidden_sizes)
        n_in = self.adjacency.shape[1]

        # sample the order of the inputs and the connectivity of all neurons
        m[-1] = np.arange(n_in)
        for layer in range(n_layers):
            r = [n_in - 1 - (i % n_in) for i in range(hidden_sizes[layer])]
            m[layer] = np.array(r)

        # construct the mask matrices
        masks = [m[i - 1][:, None] <= m[i][None, :] for i in range(n_layers)]
        masks.append(m[n_layers - 1][:, None] < m[-1][None, :])

        return masks
