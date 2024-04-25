import numpy as np

from strnn.factorizers import ZukoFactorizer

zuko_adjacency = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])
zuko_mask_0 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
])

zuko_mask_1 = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

zuko_mask_2 = np.array([
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
])


def test_strnn_factorization_zuko():
    """Test correctness of Zuko factorization algorithm."""
    factorizer = ZukoFactorizer(zuko_adjacency)
    masks = factorizer.factorize([4, 4])

    assert np.all(masks[0] == zuko_mask_0)
    assert np.all(masks[1] == zuko_mask_1)
    assert np.all(masks[2] == zuko_mask_2)
