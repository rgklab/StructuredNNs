import multiprocessing as mp
import ctypes

import numpy as np

from .greedy import GreedyFactorizer

M1_global: ctypes.c_long


class GreedyParallelFactorizer(GreedyFactorizer):
    """Implements greedy adjacency factorizer with parallelization.

    See GreedyFactorizer for description. This method parallelizes the
    operation which zeros out the M1 matrix across rows.
    """

    def __init__(self, adjacency: np.ndarray, opt_args: dict | None = None):
        """Initialize a parallel greedy algorithm mask factorizer.

        Args:
            adjacency: Global adjacency matrix to factorize.
            opt_args: Unused.
        """
        super().__init__(adjacency, opt_args)

    def _factorize_single_mask_greedy(
        self,
        adj_mtx: np.ndarray,
        n_hidden: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Factorize adj_mtx into M1 * M2 using parallelization.

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

        # Construct shared memory block representing M1
        M1_shared = mp.Array("d", adj_mtx.shape[0] * n_hidden)

        # Set to all ones
        M1_np = np.frombuffer(memoryview(M1_shared.get_obj()))
        M1_np = M1_np.reshape((adj_mtx.shape[0], n_hidden))
        M1_np[:] = 1

        def init(M1_mem_block):
            global M1_global
            M1_global = M1_mem_block

        with mp.Pool(
            initializer=init,
            initargs=(M1_shared,),
            processes=mp.cpu_count()
        ) as pool:
            pool.starmap(
                self._factorize_row,
                [(i, M2, adj_mtx) for i in range(adj_mtx.shape[0])]
            )

        M1_np = np.frombuffer(memoryview(M1_shared.get_obj()))
        M1_np = M1_np.reshape((adj_mtx.shape[0], n_hidden))

        return M1_np, M2

    def _factorize_row(
        self,
        idx: int,
        M2: np.ndarray,
        adj_mtx: np.ndarray
    ):
        """Process factorization associated with single row.

        Args:
            idx: Row index to factorize.
            M1: Mask of shape (n_outputs x n_hidden) to factorize.
            M2: Mask of shape (n_hidden x n_inputs) to factorize.
            adj_mtx: Adjacency matrix to respect.
        """
        M1_local = np.frombuffer(memoryview(M1_global.get_obj()))
        M1_local = M1_local.reshape((adj_mtx.shape[0], M2.shape[0]))

        # Find indices where A is zero on the ith row
        Ai_zero = np.where(adj_mtx[idx, :] == 0)[0]

        # find row using closed-form solution
        # find unique entries (rows) in j-th columns of M2 where Aij = 0
        row_idx = np.unique(np.where(M2[:, Ai_zero] == 1)[0])

        M1_local[idx, row_idx] = 0.0
