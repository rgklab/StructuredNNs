import numpy as np

import torch
import torch.nn as nn

from ..models import Array_like

NONLINEARITIES = {
    'tanh': nn.Tanh(),
    'softplus': nn.Softplus(),
    'relu': nn.ReLU(),
}


def cast_adj_mat(A: Array_like, out: str):
    """Helper to convert adjacency matrix between torch and numpy.
    Args:
        A: 2D Binary adjacency matrix.
        out: Type of array to output. Either ["torch" or "numpy"]
    """
    assert torch.is_tensor(A) or isinstance(A, np.ndarray)

    if out == "torch":
        if torch.is_tensor(A):
            return A
        else:
            return torch.Tensor(A)
    elif out == "np":
        if isinstance(A, np.ndarray):
            return A
        else:
            return A.numpy()
    else:
        raise ValueError("Invalid out type")
