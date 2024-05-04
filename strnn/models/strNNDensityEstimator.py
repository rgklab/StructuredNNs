# Note: for Gaussian cases, need to construct a full 2d-by-d adj mtx
# StrNN network no longer handles Gaussian case automatically

import torch
import torch.nn.functional as F
import numpy as np
from strnn.models.strNN import StrNN
from strnn.models.adaptive_layer_norm import AdaptiveLayerNorm
from numpy.random import binomial


SUPPORTED_DATA_TYPES = ['binary', 'gaussian']


class StrNNDensityEstimator(StrNN):
    def __init__(self,
        nin: int,
        hidden_sizes: tuple[int, ...],
        nout: int,
        opt_type: str = 'greedy',
        opt_args: dict = {'var_penalty_weight': 0.0},
        precomputed_masks: np.ndarray | None = None,
        adjacency: np.ndarray | None = None,
        activation: str = 'relu',
        data_type: str = 'binary',
        init_type: str = 'ian_uniform',
        norm_type: str | None = None,
        layer_norm_inverse: bool | None = None,
        init_gamma: float | None = None,
        # min_gamma: float | None = None,
        max_gamma: float | None = None,
        anneal_rate: float | None = None,
        anneal_method: str | None = None,        
        wp: float | None = None
    ):
        """
        Initialize a StrNNDensityEstimator for closed-form density estimation tasks
        
        Args:
            init_type: initialization scheme for weights
            norm_type: normalization type: layer, batch, adaptive_layer
            TODO: Update doctring
        """
        super().__init__(
            nin, hidden_sizes, nout, opt_type, opt_args,
            precomputed_masks, adjacency, activation,
            init_type, norm_type, layer_norm_inverse, 
            init_gamma, max_gamma, anneal_rate, anneal_method, wp
        )
        assert data_type in SUPPORTED_DATA_TYPES
        self.data_type = data_type


    def forward(self, x: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input data
            gamma (float): temperature parameter for adaptive layer normalization
        """
        for layer in self.net:
            if isinstance(layer, AdaptiveLayerNorm):
                x = layer(gamma, x)
            else:
                x = layer(x)
        return x
    

    def compute_LL(self, x, x_hat):
        """
        Compute negative log likelihood given input x and reconstructed x_hat
        """
        mu, log_sigma = x_hat[:, :self.nin], x_hat[:, self.nin:]
        z = (x - mu) * torch.exp(-log_sigma)
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = - log_sigma.sum(1) + log_prob_gauss

        return ll, z

    def get_preds_loss(self, batch: torch.Tensor, gamma: float | None = None):
        """
        Args:
            batch (torch.Tensor): input data
            gamma (float): temperature parameter for adaptive layer normalization
        """
        assert self.data_type in SUPPORTED_DATA_TYPES
        x = batch
        x_hat = self(x, gamma) # Pass gamma into forward function

        if self.data_type == 'binary':
            # Evaluate the binary cross entropy loss
            loss = F.binary_cross_entropy_with_logits(
                x_hat, x, reduction='sum'
            ) / len(x)
        else:
            # Assume data is Gaussian if not binary
            loss = - self.compute_LL(x, x_hat)[0].sum() / len(x)

        return x_hat, loss

    def generate_sample(self, x0):
        """
        Generate a data sample using trained model
        BINARY VERSION ONLY AT THE MOMENT!!!

        @param x0: value of first data dimension
        @return: generated sample
        """
        sample = torch.from_numpy(np.zeros(self.nin))
        sample[0] = x0

        for d in range(1, self.nin):
            out = self(sample.float())
            sig = torch.nn.Sigmoid()
            out = sig(out)
            p_d = out[d]
            x_d = binomial(1, p=p_d.detach().numpy())
            sample[d] = x_d
        return sample


if __name__ == '__main__':
    A = np.ones((6, 3))
    A = np.tril(A, -1)
    model = StrNNDensityEstimator(
        nin=3,
        hidden_sizes=(6,),
        nout=6,
        opt_type='Zuko',
        opt_args={'var_penalty_weight': 0.0},
        precomputed_masks=None,
        adjacency=A,
        activation='relu')
    print(model.A)

