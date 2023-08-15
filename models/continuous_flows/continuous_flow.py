import torch

from ..normalizing_flow import NormalizingFlow


class ContinuousFlow(NormalizingFlow):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
