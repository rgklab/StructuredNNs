import torch
from torch import nn
import numpy as np
from torch.nn.functional import softmax


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, wp=1.0, inverse=False, eps=1e-5):
        """
        Args:
            wp (float) min: 0.0, max: 1.0
                Weight parameter that determines how much the activations are 
                reweighted based on numbers of connections.
            inverse (bool): If False (default), AdaptiveLayerNorm puts emphasis 
                on nodes with more incoming connections.
                If True, AdaptiveLayerNorm puts emphasis on nodes with fewer
                incoming connections.
        """
        super().__init__()
        self.eps = eps
        self.wp = wp
        self.inverse = inverse
        self.norm_weights = None


    def set_mask(self, mask_so_far):
        """
        Args:
            mask_so_far (np.ndarray)
                The mask that has been applied from inputs to this layer, 
                used to assign weights to the nodes during normalization.
        """
        self.mask_so_far = mask_so_far
        self.connections = torch.tensor(
            mask_so_far.sum(axis=1), dtype=torch.float32)
        

    def forward(self, gamma, x: torch.Tensor) -> torch.Tensor:
        """
        gamma (float) min: 0.0, max: 1.0
            Temperature variable (starting value) in standard softmax function.
            High temperature -> smoothing effect
            Low temperature -> basically argmax/one-hot: the nodes with more incoming
                connections are more favoured.
        """
        # Reweight by number of conenctions first
        if self.connections is not None:
            self.connections = self.connections.to(x.device)
            # Pass connections through softmax with gamma as temperature
            self.norm_weights = softmax(
                self.connections / gamma, dim=0)
            # x =  self.wp * x * self.norm_weights + (1 - self.wp) * x
            if self.inverse:
                x = x / self.norm_weights
            else:
                x = x * self.norm_weights * self.mask_so_far.shape[0]
        else:
            raise ValueError("norm_weights must be set before forward pass.")
        
        # Normalize the activations x via regular layer norm second
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, keepdim = True, unbiased=False)
        y = ((x - mean) / torch.sqrt(var + self.eps))

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