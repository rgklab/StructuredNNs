import numpy as np


def generate_adj_mat_uniform(data_dim: int, threshold: float) -> np.ndarray:
    """Generate adjacency matrix with uniform sparsity.

    Args:
        data_dim: Dimension of data.
        threshold: Sparsity threshold. Higher value introduces more sparsity.
    Returns:
        2D binary adjacency matrix.
    """
    A = np.random.uniform(size=(data_dim, data_dim))
    A = (A > threshold).astype(int)
    A = np.tril(A, -1)
    return A
