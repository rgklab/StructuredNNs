from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ....models import TTuple


class Normalizer(nn.Module, metaclass=ABCMeta):
    """Interface for normalizing flow normalizer."""

    @abstractmethod
    def forward(self, x: torch.Tensor, params: torch.Tensor) -> TTuple:
        """Abstract method for normalizer forward pass."""
        pass

    @abstractmethod
    def invert(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Abstract method for normalizer reverse pass."""
        pass
