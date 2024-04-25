import multiprocessing as mp
import ctypes

import numpy as np

from .greedy import GreedyFactorizer

M1_global: ctypes.c_long
M2_global: ctypes.c_long
A_global: ctypes.c_long


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

        Matrices M2 and A only require read operations in child processes.
        M1 can be handled without synchronization as each matrix element is
        only written to once.

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

        n_out = adj_mtx.shape[0]
        n_in = adj_mtx.shape[1]

        # Construct shared memory blocks
        M1_shared = mp.RawArray("d", n_out * n_hidden)
        M2_shared = mp.RawArray("d", n_hidden * n_in)
        A_shared = mp.RawArray("d", n_out * n_in)

        # Set to all ones
        M1_np = np.frombuffer(memoryview(M1_shared))
        M1_np = M1_np.reshape((n_out, n_hidden))
        M1_np[:] = 1

        M2_np = np.frombuffer(memoryview(M2_shared))
        M2_np = M2_np.reshape((n_hidden, n_in))
        M2_np[:] = M2

        A_np = np.frombuffer(memoryview(A_shared))
        A_np = A_np.reshape((n_out, n_in))
        A_np[:] = adj_mtx

        def init(M1_mem_block, M2_mem_block, A_mem_block):
            global M1_global
            global M2_global
            global A_global

            M1_global = M1_mem_block
            M2_global = M2_mem_block
            A_global = A_mem_block

        with mp.Pool(
            initializer=init,
            initargs=(M1_shared, M2_shared, A_shared),
            processes=mp.cpu_count()
        ) as pool:
            pool.starmap(
                self._factorize_row,
                [(i, (n_in, n_hidden, n_out)) for i in range(n_out)]
            )

        M1_np = np.frombuffer(memoryview(M1_shared))
        M1_np = M1_np.reshape((adj_mtx.shape[0], n_hidden))

        return M1_np, M2

    def _factorize_row(self, idx: int, shapes: tuple[int, int, int]):
        """Process factorization associated with single row.

        Args:
            idx: Row index to factorize.
            shapes: Tuple of (n_in, n_hidden, n_out).
        """
        M1_local = np.frombuffer(memoryview(M1_global))
        M1_local = M1_local.reshape((shapes[2], shapes[1]))

        M2_local = np.frombuffer(memoryview(M2_global))
        M2_local = M2_local.reshape((shapes[1], shapes[0]))

        A_local = np.frombuffer(memoryview(A_global))
        A_local = A_local.reshape((shapes[2], shapes[0]))

        # Find indices where A is zero on the ith row
        Ai_zero = np.where(A_local[idx, :] == 0)[0]

        # find row using closed-form solution
        # find unique entries (rows) in j-th columns of M2 where Aij = 0
        row_idx = np.unique(np.where(M2_local[:, Ai_zero] == 1)[0])

        M1_local[idx, row_idx] = 0.0
