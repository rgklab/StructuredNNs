from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ....models import TTuple


class Normalizer(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, params: torch.Tensor) -> TTuple:
        pass

    @abstractmethod
    def invert(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        pass
