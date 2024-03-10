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

# Manually computed masks according to StrNN's greedy initialization
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


def test_strnn_check_masks_valid():
    """Test that valid masks result in positive check."""
    strnn = StrNN(4, [5, 5], 4, adjacency=strnn_adjacency)

    # Manually set masks for testing
    strnn.masks = [greedy_mask_0, greedy_mask_1, greedy_mask_2]

    assert strnn.check_masks() is True


def test_strnn_check_masks_valid_subopt():
    """Test that suboptimal but valid masks result in positive check."""
    strnn = StrNN(4, [5, 5], 4, adjacency=strnn_adjacency)

    # Manually set masks for testing
    subopt_mask = greedy_mask_1.copy()
    subopt_mask[0][0] = 0

    strnn.masks = [greedy_mask_0, subopt_mask, greedy_mask_2]
    assert strnn.check_masks() is True


def test_strnn_check_masks_invalid():
    """Test that invalid masks result in negative check."""
    strnn = StrNN(4, [5, 5], 4, adjacency=strnn_adjacency)

    # Manually set masks for testing
    invalid_mask = greedy_mask_1.copy()
    invalid_mask[0][1] = 1

    strnn.masks = [greedy_mask_0, invalid_mask, greedy_mask_2]
    assert strnn.check_masks() is False


def test_strnn_factorization_greedy_single_layer():
    """Test correctness of single layer greedy factorization algorithm."""
    strnn = StrNN(
        nin=5,
        hidden_sizes=[7, 7],
        nout=5,
        opt_type="greedy",
        adjacency=greedy_factor_adjacency
    )

    M1, M2 = strnn.factorize_single_mask_greedy(
        adj_mtx=greedy_factor_adjacency,
        n_hidden=7
    )

    assert np.all(M1 == greedy_factor_mask_0)
    assert np.all(M2 == greedy_factor_mask_1)


def test_strnn_factorization_greedy():
    """Test correctness of multi-layer greedy factorization algorithm."""
    strnn = StrNN(
        nin=4,
        hidden_sizes=[5, 5],
        nout=4,
        opt_type="greedy",
        adjacency=strnn_adjacency
    )

    masks = strnn.factorize_masks()

    assert np.all(masks[0] == greedy_mask_0)
    assert np.all(masks[1] == greedy_mask_1)
    assert np.all(masks[2] == greedy_mask_2)


def test_strnn_made_adjacency():
    """Test scenario where MADE factorization is used with adjacency.

    Adjacency must either be lower triangular or None.
    """
    pass


def test_strnn_no_adjacency():
    """Test scenario where no adjacency is passed.

    Adjacency should default to fully lower triangular.
    """
    strnn = StrNN(4, [4, 4], 4, opt_type="MADE", adjacency=None)

    assert np.all(strnn.A == np.tril(np.ones((4, 4)), -1))


def test_strnn_factorization_made():
    """Test correctness of MADE factorization algorithm."""
    strnn = StrNN(4, [4, 4], 4, opt_type="MADE")

    masks = strnn.factorize_masks_MADE()

    assert np.all(masks[0] == made_mask_0)
    assert np.all(masks[1] == made_mask_1)
    assert np.all(masks[2] == made_mask_2)

    # Verify validity
    strnn.masks = masks
    assert strnn.check_masks() is True


def test_strnn_factorization_zuko():
    """Test correctness of Zuko factorization algorithm."""
    strnn = StrNN(4, [4, 4], 4, opt_type="Zuko", adjacency=strnn_adjacency)

    masks = strnn.factorize_masks_zuko([4, 4])

    assert np.all(masks[0] == zuko_mask_0)
    assert np.all(masks[1] == zuko_mask_1)
    assert np.all(masks[2] == zuko_mask_2)

    strnn.masks = masks
    assert strnn.check_masks() is True


def test_strnn_update_masks():
    """Test that update masks correctly interfaces with MaskedLinear"""
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
    strnn = StrNN(4, [4, 4], 8, adjacency=strnn_adjacency)

    strnn.update_masks()

    assert strnn.masks[-1].shape == (4, 8)
