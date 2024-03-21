import numpy as np

from .factorizer import AdjacencyFactorizer


class GreedyFactorizer(AdjacencyFactorizer):
    """Implements greedy adjacency factorizor.

    Main factorization algorithm introduced by StrNN paper.

    Aims to maximize connections in resulting mask matrices. Factorization
    makes greedy assumption, which copies non-zero rows from the adjacency
    matrix into M2 during factorization. Empirically, this results in a
    factorization that has close to the optimal number of connections
    without high asymptotic cost.
    """

    def __init__(self, adjacency: np.ndarray, opt_args: dict | None = None):
        """Initialize a greedy algorithm mask factorizer.

        Args:
            adjacency: Global adjacency matrix to factorize.
            opt_args: Unused.
        """
        self.adjacency = adjacency
        self.opt_args = opt_args

    def factorize(self, hidden_sizes: tuple[int, ...]) -> list[np.ndarray]:
        """Factorize adjacency matrix into mask matrices.

        Factorize the given adjacency structure into per-layer masks.
        We use a recursive approach here for efficiency and simplicity.

        Args:
            hidden_sizes: List of hidden widths for each intermediate layer.

        Returns:
            List of masks in order for layers from inputs to outputs. This
            order matches how the masks are assigned to the networks in MADE.
        """
        masks: list[np.ndarray] = []
        adj_mtx = np.copy(self.adjacency)

        for layer in hidden_sizes:
            (M1, M2) = self._factorize_single_mask_greedy(adj_mtx, layer)

            # Update the adjacency structure for recursive call
            adj_mtx = M1

            # Take transpose for size: (n_inputs x n_hidden/n_output)
            masks = masks + [M2.T]

        masks = masks + [M1.T]

        return masks

    def _factorize_single_mask_greedy(
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
