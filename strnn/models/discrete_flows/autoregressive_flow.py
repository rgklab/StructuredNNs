import torch
import torch.nn as nn

from .conditioners import Conditioner, StrNNConditioner, GNFConditioner, \
    MADEConditioner
from .normalizers import Normalizer, AffineNormalizer, MonotonicNormalizer

from ..normalizing_flow import NormalizingFlow, NormalizingFlowFactory
from ..config_constants import *

from ...models import TTuple


class AutoregressiveFlowStep(NormalizingFlow):
    """A single normalizing flow step which can be chained together."""

    def __init__(self, conditioner: Conditioner, normalizer: Normalizer):
        """Initialize an autoregressive flow step.

        Args:
            conditioner: Flow conditioner.
            normalizer: Flow normalizer.
        """
        super().__init__()
        self.conditioner = conditioner
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor) -> TTuple:
        """Compute forward pass of flow step.

        Args:
            x: Input data.
        Returns:
            Transformed data, and jacobian determinant.
        """
        params = self.conditioner(x)
        z, jac = self.normalizer(x, params)
        return z, torch.log(jac).sum(1)

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Perform inverse transformation from latent to observed space.

        Args:
            z: Input data from latent space.
        Returns:
            Transformed data.
        """
        x = torch.zeros_like(z)

        for i in range(self.conditioner.input_dim + 1):
            params = self.conditioner(x)
            x_prev = x
            x = self.normalizer.invert(z, params)
            if torch.norm(x - x_prev) == 0.:
                break
        return x


class AutoregressiveFlow(NormalizingFlow):
    """Implements a chain of autoregressive flow steps."""

    def __init__(self, steps: list[AutoregressiveFlowStep], permute: bool):
        """Initialize AutoregressiveFlow.

        Args:
            steps: List of AutoregressiveFlowSteps to be composed.
            permute: Flips the variable order between flow steps.
        """
        super().__init__()
        self.permute = permute
        self.steps = nn.ModuleList(steps)

    def forward(self, x: torch.Tensor) -> TTuple:
        """Compute forward pass through chained steps.

        Args:
            x: Input data.

        Returns:
            Transformed data and jacobian determinant.
        """
        jac_tot = torch.zeros(x.shape[0]).to(x)

        inv_idx = torch.arange(x.shape[1] - 1, -1, -1).long()
        for step in self.steps:
            z, jac = step(x)
            jac_tot += jac

            if self.permute:
                x = z[:, inv_idx]
            else:
                x = z

        return z, jac_tot

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Inverted transform of composed flow steps.

        Args:
            z: Input data from latent space.
        Returns:
            Transformed data.
        """
        for step in range(1, len(self.steps) + 1):
            x = self.steps[-step].invert(z)

            if self.permute:
                inv_idx = torch.arange(z.shape[1] - 1, -1, -1).long()
                z = x[:, inv_idx]
            else:
                z = x

        return x

    def invert_gnf(self, z: torch.Tensor) -> torch.Tensor:
        """Perform inversion according to the GNF offical repo.

        Importantly, variables are NOT re-permuted. This leads to
        poor sample quality. The order of flow steps is also
        incorrect due to an indexing error.

        Args:
            z: Input data from latent space.
        Returns:
            Transformed data.
        """
        for step in range(len(self.steps)):
            z = self.steps[-step].invert(z)

        return z


class AutoregressiveFlowFactory(NormalizingFlowFactory):
    """Constructs AutoregressiveFlow."""

    def __init__(self, config: dict):
        """Initialize AutoregressiveFlowFactory.

        Args:
            config: Dictionary of model initialization attributes.
        """
        super().__init__(config)
        self.parse_config(config)

    def parse_config(self, config: dict):
        """Parse config and stores relevant attributes."""
        self.input_dim = config[INPUT_DIM]
        self.adj = config[ADJ]
        self.flow_steps = config[FLOW_STEPS]
        self.flow_permute = config[FLOW_PERMUTE]

        self.cond_type = config[COND_TYPE]
        self.norm_type = config[NORM_TYPE]

        self.cond_hid = config[COND_HID]
        self.cond_act = config[COND_ACT]

        self.cond_out = config[N_PARAM_PER_VAR]

        if self.cond_type == "strnn":
            self.opt_type = config[OPT_TYPE]
            self.opt_args = config[OPT_ARGS]
        elif self.cond_type == "gnf":
            self.gnf_hot = config[GNF_HOT]
        elif self.cond_type == "made":
            pass
        else:
            raise ValueError("Unknown conditioner type.")

        if self.norm_type == "affine":
            assert self.cond_out == 2
        elif self.norm_type == "umnn":
            self.umnn_int_hid = config[UMNN_INT_HID]
            self.umnn_step = config[UMNN_INT_STEP]
            self.umnn_solver = config[UMNN_INT_SOLVER]
        else:
            raise ValueError("Unknown normalizer type.")

    def _build_flow_step(self) -> AutoregressiveFlowStep:
        """Construct a single autoregressive flow step according to config.

        Returns:
            Initialized AutoregressiveFlowStep.
        """
        conditioner: Conditioner
        normalizer: Normalizer

        if self.cond_type == "strnn":
            conditioner = StrNNConditioner(
                self.input_dim,
                self.cond_hid,
                self.cond_out,
                self.cond_act,
                self.adj,
                self.opt_type,
                self.opt_args
            )
        elif self.cond_type == "made":
            conditioner = MADEConditioner(
                self.input_dim,
                self.cond_hid,
                self.cond_out,
                self.cond_act
            )
        elif self.cond_type == "gnf":
            conditioner = GNFConditioner(
                self.input_dim,
                self.cond_hid,
                self.cond_out,
                self.cond_act,
                self.gnf_hot,
                self.adj
            )
        else:
            raise ValueError("Unknown conditioner type.")

        if self.norm_type == "affine":
            normalizer = AffineNormalizer()
        elif self.norm_type == "umnn":
            normalizer = MonotonicNormalizer(
                self.umnn_int_hid,
                self.cond_out,
                self.umnn_step,
                self.umnn_solver
            )
        else:
            raise ValueError("Unknown normalizer type.")

        return AutoregressiveFlowStep(conditioner, normalizer)

    def _build_flow(self) -> AutoregressiveFlow:
        """Build AutoregressiveFlow out of composed flow steps.

        Returns:
            Composition of autoregressive flow steps.
        """
        flow_steps = [self._build_flow_step() for _ in range(self.flow_steps)]

        flow = AutoregressiveFlow(flow_steps, self.flow_permute)
        return flow
