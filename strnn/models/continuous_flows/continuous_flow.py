import torch

from ..normalizing_flow import NormalizingFlow
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


class ContinuousFlowFactory():
    """Constructs a FFJORD continuous normalizing flow."""
    def __init__(self, args):
        self.parse_args(args)

    def parse_args(self, args: dict):
        """Parses config for relevant arguments."""
        self.input_dim = args[INPUT_DIM]
        self.hidden_dim = args[ODENET_HID]
        self.lin_type = args[ODENET_LIN]
        self.act_type = args[ODENET_ACT]

        self.adj_mat = args[ADJ]
        self.opt_type = args[OPT_TYPE]
        self.opt_args = args[OPT_ARGS]

        self.divergence_fn = args[ODENET_DIV_FN]

        self.T = args[ODE_TIME]
        self.train_T = args[ODE_TRAIN_T]

        self.solver = args[ODE_SOLVER_TYPE]
        self.atol = args[ODE_SOLVER_ATOL]
        self.rtol = args[ODE_SOLVER_RTOL]
        self.step_size = args[ODE_SOLVER_STEP]

        self.flow_steps = args[FLOW_STEPS]

    def get_config(self):
        """Returns all arguments used in model construction for logging."""
        raise NotImplementedError

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

    def build_cnf(self) -> ContinuousFlow:
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
