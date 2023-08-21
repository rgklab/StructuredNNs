from abc import ABCMeta, abstractmethod

from math import pi

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Evaluates likelihood of z under standard normal."""
    return -.5 * (torch.log(torch.tensor(pi) * 2) + z ** 2).sum(1)


class NormalizingFlow(nn.Module, metaclass=ABCMeta):
    """Interface for Normalizing Flows."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class NormalizingFlowLearner(pl.LightningModule):
    def __init__(self, flow: NormalizingFlow, lr: float):
        self.flow = flow
        self.lr = lr

    def forward(self, x):
        return self.flow(x)

    def invert(self, z):
        return self.invert(z)

    def training_step(self, batch, batch_idx):
        z, jac = self.flow.forward(batch)
        logpz = standard_normal_logprob(z)

        logpx = logpz - jac
        loss = -torch.mean(logpx)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
