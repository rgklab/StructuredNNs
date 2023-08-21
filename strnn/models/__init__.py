import numpy as np
import torch

from .normalizing_flow import NormalizingFlowLearner

from typing import Union, Tuple

Array_like = Union[torch.Tensor, np.ndarray]
TTuple = Tuple[torch.Tensor, torch.Tensor]

__all__ = ["NormalizingFlowLearner"]
