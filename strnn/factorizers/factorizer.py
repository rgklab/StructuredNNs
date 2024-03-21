from abc import ABC, abstractmethod

import numpy as np


def check_masks(masks: list[np.ndarray], adjacency: np.ndarray) -> bool:
    """Check whether weight masks respect prescribed adjacency matrix.

    Given a set of masks, [M1, M2, ..., Mk], check if the matrix product
    (M1 * M2 * ... * Mk).T = Mk.T * ... * M2.T * M1.T
    respects the provided adjacency structure.

    Args:
        masks: List of masks [M1, M2, ..., Mk]
        adjacency: Adjacency matrix of shape (n_output x n_input)

    Returns:
        True or False depending on validity of weight masks.
    """
    mask_prod = masks[0]
    for i in range(1, len(masks)):
        mask_prod = mask_prod @ masks[i]
    mask_prod = mask_prod.T

    constraint = (mask_prod > 0.0001) * 1. - adjacency

    return not np.any(constraint != 0.)


class AdjacencyFactorizer(ABC):
    """Interface for adjacency factorization algorithms."""

    @abstractmethod
    def __init__(self, adjacency: np.ndarray, opt_args: dict | None = None):
        """Abstract initialization template."""
        pass

    @abstractmethod
    def factorize(
        self,
        hidden_sizes: tuple[int, ...]
    ) -> list[np.ndarray]:
        """Abstract factorization method."""
        pass
