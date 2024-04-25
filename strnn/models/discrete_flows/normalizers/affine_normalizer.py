import torch

from .normalizer import Normalizer
from ....models import TTuple


class AffineNormalizer(Normalizer):
    """Affine normalizer.

    Code taken from: https://github.com/AWehenkel/Graphical-Normalizing-Flows.
    """

    def __init__(self):
        """Initialize Affine Normalizer."""
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> TTuple:
        """Compute affine transform.

        Args:
            x: Input data.
            h: Transform parameters from conditioner. Should be a (B x D x 2)
                tensor where B is batch size, D is input dimension.

        Returns:
            Transformed data and jacobian determinant.
        """
        mu = h[:, :, 0].clamp_(-5., 5.)
        sigma = torch.exp(h[:, :, 1].clamp_(-5., 2.))

        z = x * sigma + mu
        return z, sigma

    def invert(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute inverse affine transform.

        Args:
            z: Data from latent space.
            h: Transform parameters from conditioner. Should be a (B x D x 2)
                tensor where B is batch size, D is input dimension.

        Returns:
            Inverse transformed data.
        """
        mu = h[:, :, 0].clamp_(-5., 5.)
        sigma = torch.exp(h[:, :, 1].clamp_(-5., 2.))

        x = (z - mu) / sigma
        return x
