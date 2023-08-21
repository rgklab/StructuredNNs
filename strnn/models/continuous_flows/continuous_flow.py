import torch

from ..normalizing_flow import NormalizingFlow, NormalizingFlowFactory
from ..config_constants import *

from .odenets import WeilbachSparseODENet, FCODEnet, StrODENet, ODENet

from .ffjord.cnf import CNF
from .ffjord.container import SequentialFlow
from .ffjord.odefunc import ODEfunc

from ...models import TTuple


class ContinuousFlow(NormalizingFlow):
    def __init__(self, ffjord_cnf: SequentialFlow):
        super().__init__()
        self.model = ffjord_cnf

    def forward(self, x: torch.Tensor) -> TTuple:
        return self.model(x, reverse=False)

    def invert(self, z: torch.Tensor) -> TTuple:
        return self.model(z, reverse=True)


class ContinuousFlowFactory(NormalizingFlowFactory):
    """Constructs a FFJORD continuous normalizing flow."""
    def __init__(self, config: dict):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config: dict):
        """Parses config for relevant arguments."""
        self.input_dim = config[INPUT_DIM]
        self.hidden_dim = config[ODENET_HID]
        self.lin_type = config[ODENET_LIN]
        self.act_type = config[ODENET_ACT]

        self.adj_mat = config[ADJ]
        self.opt_type = config[OPT_TYPE]
        self.opt_args = config[OPT_ARGS]

        self.divergence_fn = config[ODENET_DIV_FN]

        self.T = config[ODE_TIME]
        self.train_T = config[ODE_TRAIN_T]

        self.solver = config[ODE_SOLVER_TYPE]
        self.atol = config[ODE_SOLVER_ATOL]
        self.rtol = config[ODE_SOLVER_RTOL]
        self.step_size = config[ODE_SOLVER_STEP]

        self.flow_steps = config[FLOW_STEPS]

    def build_odenet(self) -> ODENet:
        """Constructs neural network used to model ODE dynamics function.

        Returns:
            Neural network used to parameterize ODE dynamics.
        """
        net: ODENet | None = None
        if self.lin_type == "weilbach":
            net = WeilbachSparseODENet(
                self.input_dim,
                len(self.hidden_dim),
                self.act_type,
                self.adj_mat
            )
        elif self.lin_type == "fully_connected":
            net = FCODEnet(
                self.input_dim,
                self.hidden_dim,
                self.act_type
            )
        elif self.lin_type == "strnn":
            net = StrODENet(
                self.input_dim,
                self.hidden_dim,
                self.act_type,
                self.opt_type,
                self.opt_args,
                self.adj_mat,
            )
        else:
            raise ValueError("Unknown ODENet type.")

        return net

    def build_cnf_step(self) -> CNF:
        """Builds a FFJORD continuous normalizing flow.

        TODO: Reimplement capability to regularize flows

        Returns:
            FFJORD CNF.
        """
        odenet = self.build_odenet()
        odefunc = ODEfunc(
            diffeq=odenet,
            divergence_fn=self.divergence_fn
        )
        cnf = CNF(
            odefunc=odefunc,
            T=self.T,
            train_T=self.train_T,
            solver=self.solver,
        )
        return cnf

    def build_flow(self) -> ContinuousFlow:
        """Builds a chain of continuous normalizing flows.

        Returns:
            Sequence of FFJORD CNFs wrapped in ContinuousFlow class
        """
        chain = [self.build_cnf_step() for _ in range(self.flow_steps)]
        model = SequentialFlow(chain)

        def _set(module):
            if isinstance(module, CNF):
                # Set solver settings
                module.solver = self.solver
                module.atol = self.atol
                module.rtol = self.rtol

                if self.step_size is not None:
                    module.solver_options["step_size"] = self.step_size

                # If using fixed-grid adams, restrict order to not be too high.
                if self.solver in ["fixed_adams", "explicit_adams"]:
                    module.solver_options["max_order"] = 4

        model.apply(_set)

        cnf = ContinuousFlow(model)
        return cnf
