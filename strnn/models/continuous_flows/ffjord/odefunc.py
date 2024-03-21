import torch
import torch.nn as nn

from ....models import TTuple


class DivergenceFunction():
    """Implements computations for divergence function."""

    def __init__(self, divergence_fn: str):
        """Initialize divergence function."""
        assert divergence_fn in ("brute_force", "approximate")
        self.divergence_fn = divergence_fn

    def get_divergence(
        self,
        dy: torch.Tensor,
        y: torch.Tensor,
        e: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence function."""
        if self.divergence_fn == "brute_force":
            return self.divergence_bf(dy, y)
        elif self.divergence_fn == "approximate":
            return self.divergence_approx(dy, y, e)
        else:
            raise ValueError()

    def divergence_bf(self, dx: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute brute-force divergence."""
        sum_diag = torch.zeros(dx.shape[0]).to(y)
        for i in range(y.shape[1]):
            grad = torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)
            sum_diag += grad[0].contiguous()[:, i].contiguous()
        return sum_diag.contiguous()

    def divergence_approx(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        e: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute approximate divergence."""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx


class ODEfunc(nn.Module):
    """Local copy of FFJORD's odefunc.py as FFJORD is unavailable as a package.

    https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py
    """

    def __init__(self, diffeq: nn.Module, divergence_fn: str = "approximate"):
        """Initialize ODEFunc.

        Args:
            diffeq: Network that models ODE dynamics.
            divergence_fn: Specifies ODE divergence function.
        """
        super().__init__()

        self.diffeq = diffeq
        self.div_fn = DivergenceFunction(divergence_fn)

        self._e: torch.Tensor | None = None

    def before_odeint(self, e: torch.Tensor | None = None):
        """Run actions prior to ODE integration."""
        self._e = e

    def forward(self, t: torch.Tensor, states: TTuple) -> TTuple:
        """Compute forward pass for ODE dynamics."""
        assert len(states) == 2
        y = states[0]

        # convert to tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(y)
        t = t.type_as(y)

        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y).to(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t, y)

            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence_bf = self.div_fn.divergence_bf
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.div_fn.get_divergence(dy, y, self._e)
                divergence = divergence.view(batchsize, 1)

        return (dy, -divergence)
