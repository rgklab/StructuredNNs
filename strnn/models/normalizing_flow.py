from abc import ABCMeta, abstractmethod

from math import pi

import torch
import torch.nn as nn


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Evaluate likelihood of z under standard normal.

    Args:
        z: Input data.

    Returns:
        Log probability of data under standard normal distribution.
    """
    return -.5 * (torch.log(torch.tensor(pi) * 2) + z ** 2).sum(1)


class NormalizingFlow(nn.Module, metaclass=ABCMeta):
    """Interface for Normalizing Flows."""

    config: dict

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Abstract method for NormalizingFlow forward pass."""
        pass

    @abstractmethod
    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Abstract method for NormalizingFlow reverse pass."""
        pass

    def compute_loss(self, z: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
        """Compute normalizing flow loss function.

        Computes Jacobian determinant of flow transformation summed with the
        likelihood of the transformed data under the standard normal.

        Args:
            z: Normalized data samples.
            jac: Jacobian determinant of flow transformation.

        Return:
            Normalizing flow loss of samples.
        """
        logpx = standard_normal_logprob(z) + jac
        loss = -torch.mean(logpx)
        return loss


class NormalizingFlowFactory(metaclass=ABCMeta):
    """Interface for a normalizing flow factory."""

    def __init__(self, config: dict):
        """Initialize NormalizingFlowFactory superclass.

        Args:
            config: Arguments used in flow construction.
        """
        self.config = config

    @abstractmethod
    def parse_config(self, config: dict):
        """Abstract method template for parsing model config."""
        pass

    @abstractmethod
    def _build_flow(self) -> NormalizingFlow:
        """Abstract template for concrete flow builder."""
        pass

    def build_flow(self) -> NormalizingFlow:
        """Wrap concrete flow builder to add config as an attribute."""
        flow = self._build_flow()
        flow.config = self.config

        return flow
