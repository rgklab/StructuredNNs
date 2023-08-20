"""
Local copy of FFJORD's container.py file. Select file is copied from:
https://github.com/rtqichen/ffjord/blob/master/lib/layers/container.py
as FFJORD is unavailable as a package.
"""
import torch
import torch.nn as nn

from .cnf import CNF

from ....models import TTuple


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layersList: list[CNF]):
        super().__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> TTuple:
        """Compute forward pass of stacked CNF layers.

        Modified for simplicity to always initialize logpx to zero.

        Args:
            x: Input data.
            reverse: Directionality of flow. True indicates x->z direction.
        Returns:
            Input data transformed by sequence of flows.
        """
        if reverse:
            inds = range(len(self.chain) - 1, -1, -1)
        else:
            inds = range(len(self.chain))

        logpx = torch.zeros(x.shape[0], 1).to(x)

        for i in inds:
            x, logpx = self.chain[i](x, logpx, reverse=reverse)
        return x, logpx
