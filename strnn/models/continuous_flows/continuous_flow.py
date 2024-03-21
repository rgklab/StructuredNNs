import numpy as np

import torch

from ..normalizing_flow import NormalizingFlow, NormalizingFlowFactory
from ..config_constants import *

from .odenets import WeilbachSparseODENet, FCODEnet, StrODENet, ODENet

from .ffjord.cnf import CNF
from .ffjord.container import SequentialFlow
from .ffjord.odefunc import ODEfunc

from ...models import TTuple


class ContinuousFlow(NormalizingFlow):
    """Wraps FFJORD CNF to abide to NormalizingFlow interface."""

    def __init__(self, ffjord_cnf: SequentialFlow):
        """Initialize ContinuousFlow.

        Args:
            ffjord_cnf: FFJORD CNF module.
        """
        super().__init__()
        self.model = ffjord_cnf

    def forward(self, x: torch.Tensor) -> TTuple:
        """Compute forward CNF pass.

        Args:
            x: Input data.
        Return:
            Transformed data, and jacobian determinant.
        """
        return self.model(x, reverse=False)

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Compute inverse transform from latent to data space.

        Args:
            z: Input data from latent distribution.
        Return:
            Transformed data.
        """
        return self.model(z, reverse=True)[0]


class AdjacencyModifier:
    """Helper class to modify adjacency matrix.

    Since CNFs rely on the existence of unique ODE solutions to ensure
    invertibility, we don't need a lower triangular adjacency. We can thus
    experiment with adjacencies that yield better results.

    Empirically, adding the main diagonal to adjacency is required for good
    performance, while reflecting the adjacency does not yield significant
    performance gains.
    """

    def __init__(self, modifiers: list[str]):
        """Initialize AdjacencyModifier.

        Valid modifiers are applied in input order, and be chosen from:
        ["main_diagonal", "reflect"]

        Args:
            modified: List of modifiers to apply.
        """
        self.modifiers = modifiers

    def modify_adjacency(self, adj_mat: np.ndarray) -> np.ndarray:
        """Apply all modifiers onto adjacency matrix.

        Args:
            adj_mat: 2D binary adjacency matrix.
        Returns:
            Modified adjacency matrix.
        """
        adj_mat = np.copy(adj_mat)
        for mod in self.modifiers:
            if mod == "main_diagonal":
                adj_mat = self._mod_add_diagonal(adj_mat)
            elif mod == "reflect":
                adj_mat = self._mod_reflect(adj_mat)
            else:
                raise ValueError("Modifier not implemented.")

            # Threshold matrix to be binary.
            adj_mat = (adj_mat > 0).astype(int)

        return adj_mat

    def _mod_add_diagonal(self, adj_mat: np.ndarray) -> np.ndarray:
        """Add ones to main diagonal.

        Args:
            adj_mat: 2D binary adjacency matrix.
        Returns:
            Adjacency matrix plus ones in main diagonal.
        """
        return adj_mat + np.eye(adj_mat.shape[0])

    def _mod_reflect(self, adj_mat: np.ndarray) -> np.ndarray:
        """Reflect lower triangular values to upper triangular positions.

        Args:
            adj_mat: 2D binary adjacency matrix.
        Returns:
            Adjacency matrix with ones reflected upper triangular ones.
        """
        return adj_mat + adj_mat.T


class ContinuousFlowFactory(NormalizingFlowFactory):
    """Constructs a FFJORD continuous normalizing flow."""

    def __init__(self, config: dict):
        """Initialize ContinuousFlowFactory.

        Args:
            config: ContinuousFlow architecture parameters.
        """
        super().__init__(config)
        self.parse_config(config)

    def parse_config(self, config: dict):
        """Parse config for relevant arguments.

        Args:
            config: ContinuousFlow architecture parameters.
        """
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
        """Construct neural network used to model ODE dynamics function.

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
        """Build a FFJORD continuous normalizing flow.

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

    def _build_flow(self) -> ContinuousFlow:
        """Build a chain of continuous normalizing flows.

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
