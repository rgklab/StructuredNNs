import numpy as np


def generate_adj_mat_uniform(dim: int, threshold: float) -> np.ndarray:
    """Generates adjacency matrix with uniform sparsity.

    Args:
        dim: Dimension of data.
        threshold: Sparsity threshold. Higher value introduces more sparsity.
    Returns:
        2D binary adjacency matrix.
    """
    A = np.random.uniform(size=(dim, dim))
    A = (A > threshold).astype(int)
    A = np.tril(A, -1)
    return A
