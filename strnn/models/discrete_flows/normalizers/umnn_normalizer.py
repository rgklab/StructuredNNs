
import torch
import torch.nn as nn

from UMNN import NeuralIntegral, ParallelNeuralIntegral

from .normalizer import Normalizer
from ....models import TTuple


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class ELUPlus(nn.Module):
    """Implement ELUPlus activation.

    Combines an ELU activation with a positive constant term.
    """

    def __init__(self):
        """Initialize ELUPlus."""
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ELUPlus activation.

        Args:
            x: Input logits.

        Returns:
            Activated output.
        """
        return self.elu(x) + 1.05


class IntegrandNet(nn.Module):
    """Neural network which represents an integrand for use in UMNN.

    Code taken from: https://github.com/AWehenkel/Graphical-Normalizing-Flows.
    """

    def __init__(self, hidden_dim: tuple[int], n_param_per_dim: int):
        """Initialize IntegrandNet.

        Args:
            hidden_dim: Tuple of hidden widths of network.
            n_param_per_dim: Input dimension, number of parameters per
                dimension outputted by an upstream flow conditioner.
        """
        super().__init__()
        l1 = [1 + n_param_per_dim] + list(hidden_dim)
        l2 = list(hidden_dim) + [1]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute IntegrandNet forward pass.

        Args:
            x: Input data.
            h: Parameters from conditioner network.

        Returns:
            Transformed data.
        """
        nb_batch, in_d = x.shape
        x = torch.cat((x, h), 1).view(nb_batch, -1, in_d).transpose(1, 2)

        x_he = x.contiguous().view(nb_batch * in_d, -1)
        y = self.net(x_he).view(nb_batch, -1)
        return y


class MonotonicNormalizer(Normalizer):
    """Monotonic normalizer which uses UMNN to transform data.

    Code taken from: https://github.com/AWehenkel/Graphical-Normalizing-Flows.
    """

    def __init__(
        self,
        integrand_hidden: tuple[int],
        n_param_per_dim: int,
        nb_steps: int,
        solver: str
    ):
        """Initialize MonotonicNormalizer.

        Args:
            integrand_hidden: Hidden widths of integrand network.
            n_param_per_dim: Number of conditioner parameters per variable.
            nb_steps: Number of solver steps used in neural integration.
            solver: Type of solver used. Valid: ["CC", "CCParallel"]
        """
        super().__init__()

        self.integrand_net = IntegrandNet(integrand_hidden, n_param_per_dim)
        self.nb_steps = nb_steps

        solver_dict = {
            "CC": NeuralIntegral,
            "CCParallel": ParallelNeuralIntegral,
        }
        self.solver = solver_dict[solver]

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> TTuple:
        """Perform monotonic transform.

        Args:
            x: Input data from data space.
            h: Parameters from conditioner.
        Returns:
            Transformed data and jacobian determinant.
        """
        x0 = torch.zeros(x.shape).to(x)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

        z = z0 + self.solver.apply(x0, xT, self.integrand_net,
                                   _flatten(self.integrand_net.parameters()),
                                   h, self.nb_steps)

        return z, self.integrand_net(x, h)

    def invert(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Perform inverse monotonic transform.

        Args:
            z: Input data from latent space.
            h: Parameters from conditioner.
        Returns:
            Inverse transformed data.
        """
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h)
        z_min, _ = self.forward(x_min, h)

        for i in range(20):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min
        return (x_max + x_min) / 2
