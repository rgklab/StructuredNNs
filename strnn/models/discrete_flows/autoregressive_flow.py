import torch
import torch.nn as nn

from ..normalizing_flow import NormalizingFlow


class AutoregressiveFlow(NormalizingFlow):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class AutoregressiveFlowStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class AutoregressiveFlowFactory():

    def create_flow(self, model_args: dict) -> AutoregressiveFlow:
        pass
