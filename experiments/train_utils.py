import numpy as np
from sklearn import metrics

from pytorch_lightning import Callback
from pytorch_lightning import Trainer

from strnn.models.normalizing_flow import NormalizingFlowLearner


def compute_sample_mmd(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Computes the maximal mean discrepancy between samples from X and Y.

    Args:
        x: samples from distribution X.
        y: samples from distribution Y.
        gamma: Bandwidth parameter.

    Returns:
        The MMD between the sampled distributions.
    """
    XX = metrics.pairwise.rbf_kernel(x, x, gamma)
    YY = metrics.pairwise.rbf_kernel(y, y, gamma)
    XY = metrics.pairwise.rbf_kernel(x, y, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


class CallbackComputeMMD(Callback):
    def __init__(self, n_samples: int, gamma: float):
        super().__init__()
        self.n_samples = n_samples
        self.gamma = gamma

    def on_validation_epoch_end(self, t: Trainer, nf: NormalizingFlowLearner):
        if t.val_dataloaders is None:
            msg = "Validation dataloader must be specified to compute MMD."
            raise AttributeError(msg)
        else:
            val_dl = t.val_dataloaders
            batch_size = val_dl.batch_size

        x_samples = []
        for _ in range(self.n_samples // batch_size):
            samples = nf.sample(batch_size).cpu().numpy()
            x_samples.append(samples)

        all_x_samples = np.concatenate(x_samples)
        val_dataset_np = val_dl.dataset.cpu().numpy()

        mmd = compute_sample_mmd(all_x_samples, val_dataset_np, self.gamma)
        nf.log("val_mmd", mmd)
