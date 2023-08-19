from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class NormalizingFlow(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
