import numpy as np

from strnn.factorizers import GreedyFactorizer


greedy_factor_adjacency = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0]
])

greedy_factor_mask_0 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1]
])

greedy_factor_mask_1 = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
])

greedy_adjacency = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])

greedy_mask_0 = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
])

greedy_mask_1 = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
])

greedy_mask_2 = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
])


def test_greedy_fac_single_layer():
    """Test correctness of single layer greedy factorization algorithm."""
    greedy_fac = GreedyFactorizer(None)

    M1, M2 = greedy_fac._factorize_single_mask_greedy(
        adj_mtx=greedy_factor_adjacency,
        n_hidden=7
    )

    assert np.all(M1 == greedy_factor_mask_0)
    assert np.all(M2 == greedy_factor_mask_1)


def test_greedy_fac():
    """Test correctness of multi-layer greedy factorization algorithm."""
    greedy_fac = GreedyFactorizer(greedy_adjacency)

    masks = greedy_fac.factorize([5, 5])

    assert np.all(masks[0] == greedy_mask_0)
    assert np.all(masks[1] == greedy_mask_1)
    assert np.all(masks[2] == greedy_mask_2)
