import torch
from torch import nn


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, keepdim = True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)
    

if __name__ == '__main__':
    x = torch.tensor([[1.5,.0,.0,.0]])
    layerNorm = torch.nn.LayerNorm(4, elementwise_affine = False)
    y1 = layerNorm(x)
    model = AdaptiveLayerNorm()
    y2 = model(x)

    print(y1)
    print(y2)