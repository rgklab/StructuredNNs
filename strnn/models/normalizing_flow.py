from abc import ABCMeta, abstractmethod

from math import pi

import torch
import torch.nn as nn
import torch.optim as optim

from strnn.models.config_constants import INPUT_DIM


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Evaluates likelihood of z under standard normal."""
    return -.5 * (torch.log(torch.tensor(pi) * 2) + z ** 2).sum(1)


class NormalizingFlow(nn.Module, metaclass=ABCMeta):
    """Interface for Normalizing Flows."""
    config: dict

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def invert(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def compute_loss(self, z: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
        logpx = standard_normal_logprob(z) + jac
        loss = -torch.mean(logpx)
        return loss


class NormalizingFlowFactory(metaclass=ABCMeta):
    """Interface for a normalizing flow factory."""
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def parse_config(self, config: dict):
        pass

    @abstractmethod
    def _build_flow(self) -> NormalizingFlow:
        pass

    def build_flow(self) -> NormalizingFlow:
        """Wraps concrete flow builder to add config as an attribute."""
        flow = self._build_flow()
        flow.config = self.config

        return flow
