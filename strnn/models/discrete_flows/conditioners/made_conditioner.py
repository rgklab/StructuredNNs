import numpy as np
import torch
import torch.nn as nn

from strnn import MaskedLinear
from strnn.models.model_utils import NONLINEARITIES
from .conditioner import Conditioner


class MADEConditioner(Conditioner):
    """Normalizing Flow Conditioner using MADE to compute flow parameters.

    Class is implemented using code from https://github.com/piomonti/carefl.
    This is done instead of using StrNN with "MADE" factorization to ensure
    results are directly comparable with CAREFL, but practically using
    StrNN with MADE factorization should result in the same output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int],
        n_out_param: int,
        act_type: str,
        num_masks: int = 1,
        natural_ordering: bool = True
    ):
        """Initialize an MADE conditioner.

        Args:
            input_dim: Dimension of input.
            hidden_dim: List of hidden widths for each layer.
            n_out_param: Number of output parameters per input variable.
            act_type: Activation function used in MADE.
            num_masks: Can be used to train ensemble over orderings/connections
            natural_ordering:
                Force natural ordering of dimensions and don't use random
                permutations.
        """
        super().__init__()
        self.input_dim = input_dim
        nout = input_dim * n_out_param
        self.nout = nout
        self.hidden_dim = hidden_dim
        assert self.nout % self.input_dim == 0

        # Define activation
        try:
            self.activation = NONLINEARITIES[act_type]
        except ValueError:
            raise ValueError(f"{act_type} is not a valid activation!")

        # define a simple MLP neural net
        self.module_list = []
        hs = [input_dim] + list(hidden_dim) + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.module_list.extend([
                MaskedLinear(h0, h1),
                self.activation,
            ])
        self.module_list.pop()  # pop the last activation for the output layer
        self.net = nn.Sequential(*self.module_list)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m: dict[int, np.ndarray] = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        """Apply MADE masks to masked linear layers."""
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        # bool(self.m) == False if m == {} else True
        L = len(self.hidden_dim)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = \
            np.arange(self.input_dim) \
            if self.natural_ordering else rng.permutation(self.input_dim)
        for i in range(L):
            self.m[i] = rng.randint(self.m[i - 1].min(),
                                    self.input_dim - 1,
                                    size=self.hidden_dim[i])

        # construct the mask matrices
        masks = [self.m[i - 1][:, None] <= self.m[i][None, :]
                 for i in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = input_dim * k, for integer k > 1
        if self.nout > self.input_dim:
            k = int(self.nout / self.input_dim)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [layer for layer in self.net.modules()
                  if isinstance(layer, MaskedLinear)]
        for layer, m in zip(layers, masks):
            layer.set_mask(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MADE conditioner forward pass.

        Args:
            x: Input data

        Returns:
            MADE conditioner output
        """
        return self.net(x).view(x.shape[0], -1, x.shape[1]).permute(0, 2, 1)
