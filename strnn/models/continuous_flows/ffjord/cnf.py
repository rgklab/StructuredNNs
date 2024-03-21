import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint  # type: ignore

from ....models import TTuple
from .odefunc import ODEfunc


class CNF(nn.Module):
    """Local copy of FFJORD's cnf.py as FFJORD is unavailable as a package.

    See: https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py
    """

    sqrt_end_time: torch.Tensor

    def __init__(
        self,
        odefunc: ODEfunc,
        T: float = 1.0,
        train_T: bool = False,
        solver: str = 'dopri5',
        atol: float = 1e-5,
        rtol: float = 1e-5
    ):
        """Initialize a FFJORD CNF.

        Args:
            odefunc: Dynamics function.
            T: End time of integration.
            train_T: Treats T as a trainable parameter.
            solver: ODE solver to use.
            atol: ODE solver absolute tolerance.
            rtol: ODE solver relative tolerance.
        """
        super().__init__()
        if train_T:
            t_param = nn.Parameter(torch.sqrt(torch.tensor(T)))
            self.register_parameter("sqrt_end_time", t_param)
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol

    def forward(
        self,
        z: torch.Tensor,
        logpz: torch.Tensor | None = None,
        reverse: bool = False
    ) -> TTuple:
        """Compute forward pass of CNF.

        Args:
            z: Input data.
            logpz: Accumulated jacobian term.
            reverse: Flips times of integration.
        """
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        times = [0.0, self.sqrt_end_time * self.sqrt_end_time]
        integration_times = torch.tensor(times).to(z)

        if reverse:
            integration_times = torch.flip(integration_times, (0,))

        self.odefunc.before_odeint()

        state_t = odeint(
            self.odefunc,
            (z, _logpz),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            options={},
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        return z_t, logpz_t
