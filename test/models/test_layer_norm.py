import pytest
import torch
import numpy as np

from strnn import StrNN, AdaptiveLayerNorm


def test_fully_connected_layer_norm():
    """
    When the network is fully connected, AdaptiveLayerNorm should be identical to the
    default pytorch LayerNorm (scaled by a factor of 1/h_dim).
    """
    h_dim = 8
    batch_size = 10
    x = torch.tensor(
        np.random.randn(batch_size, h_dim), dtype=torch.float32
    )
    layer_norm = torch.nn.LayerNorm(h_dim, elementwise_affine = False)
    y1 = layer_norm(x)

    input_dim = 4
    adapt_layer_norm = AdaptiveLayerNorm(gamma=1.0)
    mask_so_far = np.ones((h_dim, input_dim))
    adapt_layer_norm.set_norm_weights(mask_so_far)
    y2 = adapt_layer_norm(x)

    # y1 and y2 * h_dim should be identical
    assert torch.allclose(y1, y2 * h_dim, atol=1e-6)

    