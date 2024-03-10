import numpy as np

from strnn.factorizers import MADEFactorizer


made_adjacency = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0]
])


made_mask_0 = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0]
])

made_mask_1 = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])

made_mask_2 = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1]
])


def test_strnn_factorization_made():
    """Test correctness of MADE factorization algorithm."""
    factorizer = MADEFactorizer(made_adjacency)
    masks = factorizer.factorize([4, 4])

    assert np.all(masks[0] == made_mask_0)
    assert np.all(masks[1] == made_mask_1)
    assert np.all(masks[2] == made_mask_2)
