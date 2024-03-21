import pytest
from unittest.mock import patch

import numpy as np

from strnn import StrNN


strnn_adjacency = np.array([
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


def test_strnn_empty_adjacency():
    """Test scenario where non-MADE factorization is used with adjacency."""
    with pytest.raises(ValueError):
        StrNN(4, [4, 4], 4, opt_type="greedy", adjacency=None)


def test_strnn_empty_adjacency_made():
    """Test scenario where no adjacency is passed but opt_type is MADE.

    Adjacency should default to fully lower triangular.
    """
    strnn = StrNN(4, [4, 4], 4, opt_type="MADE", adjacency=None)

    assert np.all(strnn.A == np.tril(np.ones((4, 4)), -1))


def test_strnn_precomputed_erroneous():
    """Test that StrNN rejects precomputed masks that are incorrect."""
    incorrect_masks = [
        np.ones_like(strnn_adjacency),
        np.ones_like(strnn_adjacency)
    ]

    with pytest.raises(AssertionError):
        StrNN(
            nin=4,
            hidden_sizes=[4, 4],
            nout=4,
            adjacency=strnn_adjacency,
            precomputed_masks=incorrect_masks
        )


def test_strnn_update_masks():
    """Test that update masks correctly interfaces with MaskedLinear."""
    strnn = StrNN(4, [4, 4], 4, adjacency=strnn_adjacency)

    with patch("strnn.models.strNN.MaskedLinear.set_mask") as mock_method:
        strnn.update_masks()

        assert mock_method.call_count == 3


def test_strnn_update_masks_mismatch_output():
    """Test that update masks will enforce correct adjacency dimensions."""
    strnn = StrNN(4, [4, 4], 4, adjacency=strnn_adjacency)

    with pytest.raises(AssertionError):
        strnn.nout = 5
        strnn.update_masks()


def test_strnn_update_masks_multiple():
    """Test that update masks can use n_out that is a multiple of adj shape."""
    strnn = StrNN(4, [5, 5], 8, adjacency=strnn_adjacency)

    strnn.update_masks()

    tiled_mask = np.concatenate([greedy_mask_2] * 2, axis=1)

    assert np.all(strnn.masks[-1] == tiled_mask)
    assert np.all(strnn.net[-1].mask.numpy().T == tiled_mask)


def test_strnn_factorization_greedy():
    """Test correctness of multi-layer greedy factorization algorithm.

    Test duplicated to ensure no regression compared to pre-refactor StrNN.
    """
    strnn = StrNN(
        nin=4,
        hidden_sizes=[5, 5],
        nout=4,
        opt_type="greedy",
        adjacency=strnn_adjacency
    )

    assert np.all(strnn.masks[0] == greedy_mask_0)
    assert np.all(strnn.masks[1] == greedy_mask_1)
    assert np.all(strnn.masks[2] == greedy_mask_2)


def test_strnn_factorization_made():
    """Test correctness of MADE factorization algorithm.

    Test duplicated to ensure no regression compared to pre-refactor StrNN.
    """
    strnn = StrNN(4, [4, 4], 4, opt_type="MADE")

    assert np.all(strnn.masks[0] == made_mask_0)
    assert np.all(strnn.masks[1] == made_mask_1)
    assert np.all(strnn.masks[2] == made_mask_2)


def test_strnn_factorization_zuko():
    """Test correctness of Zuko factorization algorithm.

    Test duplicated to ensure no regression compared to pre-refactor StrNN.
    """
    strnn = StrNN(4, [4, 4], 4, opt_type="Zuko", adjacency=strnn_adjacency)

    assert np.all(strnn.masks[0] == zuko_mask_0)
    assert np.all(strnn.masks[1] == zuko_mask_1)
    assert np.all(strnn.masks[2] == zuko_mask_2)
