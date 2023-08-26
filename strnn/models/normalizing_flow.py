from abc import ABCMeta, abstractmethod

from math import pi

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

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


class NormalizingFlowLearner(pl.LightningModule):
    """PyTorch-Lightning wrapper for NormalizingFlows."""
    device: torch.device

    def __init__(self, flow: NormalizingFlow, lr: float):
        super().__init__()
        self.flow = flow
        self.lr = lr

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.flow.forward(x)

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow.invert(z)

    def sample(self, n_sample: int) -> torch.Tensor:
        """Transforms samples from latent to learned data distribution.

        Args:
            n_sample: Number of samples to generate.
        Returns:
            Sampled points from the learned data distribution.
        """
        out_dim = (n_sample, self.flow.config[INPUT_DIM])
        z_samples = torch.normal(0, 1, size=out_dim, device=self.device)
        x_samples = self.invert(z_samples)

        return x_samples

    def training_step(self, batch: torch.Tensor, idx: int) -> torch.Tensor:
        z, jac = self.flow.forward(batch)
        logpz = standard_normal_logprob(z)

        logpx = logpz - jac
        loss = -torch.mean(logpx)

        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch: torch.Tensor, idx: int) -> torch.Tensor:
        z, jac = self.flow.forward(batch)
        logpz = standard_normal_logprob(z)

        logpx = logpz - jac
        loss = -torch.mean(logpx)

        self.log("val_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.flow.parameters(), lr=self.lr)
        return optimizer
