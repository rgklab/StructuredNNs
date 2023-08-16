import torch
import torch.nn as nn

from ..strNN import StrNN
from ..normalizing_flow import NormalizingFlow


class StrODENet(StrNN):
    def forward(self, t, x):
        return super().forward(x)


class ContinuousFlow(NormalizingFlow):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def invert(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class ContinuousFlowFactory():

    def build_odenet(self,
                     input_dim: int,
                     hidden_dim: dict,
                     adj_mat: torch.Tensor,
                     opt_type: str) -> nn.Module:
        """Constructs neural network used to model ODE dynamics function.

        TODO
        Args:
            args: Model arguments.
            adj_mat: Binary adjacency matrix of shape D x D.

        Returns:
            nn.Module: NN represent dynamics function.
        """
        odenet = None
        return odenet

    def build_cnf(self, args: dict, adj_mat: torch.Tensor) -> ContinuousFlow:
        odenet = self.build_odenet(args["input_dim"], args["hidden_dim"], adj_mat)
           
        odefunc = layers.ODEfunc(
            diffeq=odenet,
            divergence_fn=args.divergence_fn
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
        )

        chain = [build_cnf() for _ in range(args.num_blocks)]
        model = layers.SequentialFlow(chain)

        set_cnf_options(args, model)

        return model
