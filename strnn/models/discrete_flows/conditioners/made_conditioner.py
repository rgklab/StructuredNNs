"""
MADE: Masked Autoencoder for Distribution Estimation
https://arxiv.org/abs/1502.03509
"""
import numpy as np
import torch.nn as nn

from strnn import MaskedLinear
from strnn.models.model_utils import NONLINEARITIES
from .conditioner import Conditioner


class MADEConditioner(Conditioner):
    """
    Normalizing Flow Conditioner using MADE to compute flow parameters.
    Code based on https://github.com/piomonti/carefl.
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
        """Initializes an MADE conditioner.

        Args:
            input_dim: Dimension of input.
            hidden_dim: List of hidden widths for each layer.
            n_out_param: Number of output parameters per input variable.
            act_type: Activation function used in MADE.
            num_masks: can be used to train ensemble over orderings/connections
            natural_ordering: force natural ordering of dimensions,
                            don't use random permutations
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
        self.net = []
        hs = [input_dim] + hidden_dim + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                self.activation,
            ])
        self.net.pop()  # pop the last activation for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
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

    def forward(self, x):
        return self.net(x)
