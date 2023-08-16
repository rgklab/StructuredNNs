import torch
import torch.nn as nn


class MaskedLinear(nn.Module):
    def __init__(self):
        pass

    def forward(self, x) -> torch.Tensor:
        pass


class StrNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: list[int],
                 out_dim: int,
                 activation: str,
                 opt_type: str,
                 opt_args: dict):
        pass

    def forward():
        pass
