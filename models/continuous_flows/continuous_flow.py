import torch
import torch.nn as nn

from ..normalizing_flow import NormalizingFlow

from .odenets import WeilbachSparseODENet, FCODEnet, StrODENet

from .ffjord.cnf import CNF
from .ffjord.container import SequentialFlow
from .ffjord.odefunc import ODEfunc


class ContinuousFlow(NormalizingFlow):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class ContinuousFlowFactory():
    """Constructs a FFJORD continuous normalizing flow."""

    def build_odenet(self,
                     net_type: str,
                     input_dim: int,
                     hidden_dim: list[int],
                     act_type: str,
                     adj_mat: torch.Tensor | None = None,
                     opt_type: str | None = None,
                     opt_args: dict | None = None) -> nn.Module:
        """Constructs neural network used to model ODE dynamics function

        Args:
            args: Model arguments.
            adj_mat: Binary adjacency matrix of shape D x D.

        Returns:
            nn.Module: NN represent dynamics function.
        """
        if net_type == "weilbach":
            n_layer = len(hidden_dim)
            net = WeilbachSparseODENet(input_dim, adj_mat, n_layer, act_type)
        elif net_type == "fully_conn":
            net = FCODEnet(input_dim, hidden_dim, act_type)
        elif net_type == "strnn":
            net = StrODENet(input_dim,
                            hidden_dim[:-1],
                            hidden_dim[-1],
                            opt_type,
                            opt_args,
                            adjacency=adj_mat,
                            activation=act_type)
        else:
            raise ValueError("Unknown ODENet type.")

        return net

    def build_cnf_step(self, args: dict, adj_mat: torch.Tensor) -> CNF:
        odenet = self.build_odenet(
            args["odenet_type"],
            args["input_dim"],
            args["hidden_dim"],
            args["act_type"],
            adj_mat,
            args["opt_type"],
            args["opt_args"])

        odefunc = ODEfunc(
            diffeq=odenet,
            divergence_fn=args["divergence_fn"]
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args["time_length"],
            train_T=args["train_T"],
            solver=args["solver"],
        )
        return cnf

    def build_cnf(self, args: dict, adj_mat: torch.Tensor) -> ContinuousFlow:

        chain = [self.build_cnf() for _ in range(args["n_steps"])]
        model = SequentialFlow(chain)

        set_cnf_options(args, model)

        return model


def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual
    model.apply(_set)
