import numpy as np

from strnn.factorizers import check_masks


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


def test_strnn_check_masks_valid():
    """Test that valid masks result in positive check."""
    masks = [greedy_mask_0, greedy_mask_1, greedy_mask_2]
    adjacency = [strnn_adjacency]

    assert check_masks(masks, adjacency) is True


def test_strnn_check_masks_valid_subopt():
    """Test that suboptimal but valid masks result in positive check."""
    subopt_mask = greedy_mask_1.copy()
    subopt_mask[0][0] = 0

    masks = [greedy_mask_0, subopt_mask, greedy_mask_2]
    adjacency = [strnn_adjacency]

    assert check_masks(masks, adjacency) is True


def test_strnn_check_masks_invalid():
    """Test that invalid masks result in negative check."""
    invalid_mask = greedy_mask_1.copy()
    invalid_mask[0][1] = 1

    masks = [greedy_mask_0, invalid_mask, greedy_mask_2]
    adjacency = [strnn_adjacency]

    assert check_masks(masks, adjacency) is False
