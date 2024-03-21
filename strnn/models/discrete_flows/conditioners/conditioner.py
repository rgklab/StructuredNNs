from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class Conditioner(nn.Module, metaclass=ABCMeta):
    """Interface for Normalizing Flow Conditioner."""

    input_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract method for forward pass."""
        pass
