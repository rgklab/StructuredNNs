import torch
from torch import nn
import numpy as np
from torch.nn.functional import softmax


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, gamma, wp, eps=1e-5):
        """
        Args:
            gamma (float) min: 0.0, max: 1.0
                Temperature variable in standard softmax function.
                High temperature -> smoothing effect
                Low temperature -> basically argmax: the nodes with more incoming
                    connections are more favoured.
            wp (float) min: 0.0, max: 1.0
                Weight parameter that determines how much the activations are 
                reweighted based on numbers of connections.
        """
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.wp = wp
        self.norm_weights = None

    def set_norm_weights(self, mask_so_far):
        """
        Args:
            mask_so_far (np.ndarray)
                The mask that has been applied from inputs to this layer, 
                used to assign weights to the nodes during normalization.
        """
        self.norm_weights = torch.tensor(
            mask_so_far.sum(axis=1), dtype=torch.float32)
        # Pass norm_weights through softmax with gamma as temperature
        self.norm_weights = softmax(
            self.norm_weights / self.gamma, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, keepdim = True, unbiased=False)
        
        # Layer norm as usual
        y = ((x - mean) / torch.sqrt(var + self.eps)) 
        
        # Reweight by number of connections if necessary
        if self.norm_weights is not None:
            self.norm_weights = self.norm_weights.to(y.device)
            y =  self.wp * y * self.norm_weights + (1 - self.wp) * y
        return y
    

if __name__ == '__main__':
    # h_dim = 8
    # batch_size = 10
    # x = torch.tensor(
    #     np.random.randn(batch_size, h_dim), dtype=torch.float32
    # )
    # layerNorm = torch.nn.LayerNorm(h_dim, elementwise_affine = False)
    # y1 = layerNorm(x)

    # model = AdaptiveLayerNorm(gamma=1.0)
    # mask_so_far = np.array([
    #    [3., 0., 0., 0.],
    #    [0., 3., 3., 0.],
    #    [3., 0., 0., 0.],
    #    [0., 3., 3., 0.],
    #    [3., 0., 0., 0.],
    #    [0., 3., 3., 0.],
    #    [3., 0., 0., 0.],
    #    [0., 3., 3., 0.]
    # ])
    # model.set_norm_weights(mask_so_far)
    # y2 = model(x)

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

    print(y1)
    print(y2 * h_dim)