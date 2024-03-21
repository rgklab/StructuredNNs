import numpy as np
from numpy.random import normal, uniform

import torch
from torch.utils.data import Dataset

from typing import Callable


class LinAddSEM:
    """Defines a linear additive SEM and sampling operations."""

    def __init__(
        self,
        noise_mean: np.ndarray,
        noise_stds: np.ndarray,
        adj_mat: np.ndarray,
        noise_dist: Callable = normal
    ):
        """Initialize Linear Additive SEM.

        Assumes autoregressive causal ordering, meaning adjacency matrix must
        be lower triangular.

        Args:
            noise_mean: Means of the noise distributions. Has shape (D,).
            noise_stds: Standard dev. of noise distributions. Has shape (D,).
            adj_mat: Adjacency matrix of SEM. Has shape (D, D).
            noise_dist: Noise generating distribution
        """
        # Check that SEM specification is valid
        assert len(noise_mean) == len(noise_stds)

        assert len(adj_mat.shape) == 2
        assert adj_mat.shape[0] == adj_mat.shape[1]
        assert len(noise_mean) == adj_mat.shape[0]
        assert np.allclose(adj_mat, np.tril(adj_mat))

        self.n_var = len(noise_mean)

        self.noise_mean = noise_mean
        self.noise_stds = noise_stds
        self.noise_dist = noise_dist
        self.adj_mat = adj_mat

    def generate_sample(self) -> np.ndarray:
        """Generate a sample from specified SEM.

        Returns:
            Single sample generated from SEM
        """
        e = self.noise_dist(self.noise_mean, self.noise_stds)

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat

    def generate_samples(self, n_samp: int) -> np.ndarray:
        """Generate multiple samples from SEM.

        Returns:
            Samples from SEM in shape (n_samp, n_dim)
        """
        return np.array([self.generate_sample() for _ in range(n_samp)])

    def generate_intervention(self, int_val: list[float | None]) -> np.ndarray:
        """Generate ground truth intervention on variables.

        Args:
            int_val:
                Intervenational values. Has shape (D,), but each position can
                be None to indicate no intervention in specific variable.

        Returns:
            Sample generated under intervention
        """
        e = self.noise_dist(self.noise_mean, self.noise_stds)

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            if int_val[i] is not None:
                out_mat[i] = int_val[i]
            else:
                out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat

    def generate_int_dist(
        self,
        int_val: list[float | None],
        n_samp: int,
        return_mean: bool = True
    ) -> np.ndarray:
        """Estimate ground truth interventional distribution.

        Generates samples under intervention, and optionally returns mean.

        Args:
            int_val:
                Intervenational values. Has shape (D,), but each position can
                be None to indicate no intervention in specific variable.
            n_samp: Number of samples used to estimate distribution
            return_mean: Whether mean of samples should be taken

        Returns:
            Interventional samples, or mean of interventional samples
        """
        samples = []
        for _ in range(n_samp):
            samples.append(self.generate_intervention(int_val))

        if return_mean:
            return np.mean(np.array(samples), axis=0)
        else:
            return np.array(samples)

    def generate_ctf_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate an sample from specified SEM, return both obs and noise.

        Return of noise that generated sample can be used to generate ground
        truth values for counterfactual queries.

        Returns:
            Sample from SEM, and noise that generated sample
        """
        e = self.noise_dist(self.noise_mean, self.noise_stds)

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat, e

    def generate_counterfactual(
        self,
        e: np.ndarray,
        ctf_val: list[float | None]
    ) -> np.ndarray:
        """Generate ground truth counterfactual outcome.

        Args:
            e: Noise used to generate original sample of interest
            ctf_val:
                Counterfactual values. Has shape (D,), but each position can
                be None to indicate no counterfactual in specific variable.

        Returns:
            Counterfactual outcome of sample
        """
        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            if ctf_val[i] is not None:
                out_mat[i] = ctf_val[i]
            else:
                out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat

    def get_carefl_ds(
        self,
        n_samp: int
    ) -> tuple[np.ndarray, None, np.ndarray]:
        """Generate dataset using SEM.

        Args:
            n_samp: Number of samples in dataset

        Returns:
            Samples, unused, and adjacency matrix
        """
        X = self.generate_samples(n_samp)

        return X, None, self.get_adj_mat()

    def get_adj_mat(self) -> np.ndarray:
        """Return adjacency matrix associated with SEM."""
        bin_adj_mat = (self.adj_mat != 0).astype(int)
        np.fill_diagonal(bin_adj_mat, 0)
        return bin_adj_mat


class RandomSEM(LinAddSEM):
    """Initializes a LinAddSEM with randomly sampled coefficients."""

    def __init__(
        self,
        dimension: int,
        noise_mean_param: tuple[float, float] = (-2, 2),
        noise_std_param: tuple[float, float] = (1, 10),
        adj_gen_param: tuple[float, float] = (-2, 2)
    ):
        """Initialize SEM with random coefficients.

        Args:
            dimension: Size of the graph
            noise_mean_param: Parameters to generate noise mean
            noise_std_param: Parameters to generate noise std
            adj_gen_param: Parameters to generate adjacency weight matrix
        """
        self.n_var = dimension

        # NOTE: Numerical issues occur if means are not zero.
        self.noise_means = uniform(*noise_mean_param, size=dimension)
        self.noise_stds = uniform(*noise_std_param, size=dimension)

        adj_mat = uniform(*adj_gen_param, size=(dimension, dimension))
        self.adj_mat = np.tril(adj_mat)

        super().__init__(self.noise_means, self.noise_stds, self.adj_mat)


class SparseSEM(LinAddSEM):
    """Initializes a LinAddSEM with many independencies."""

    def __init__(
        self,
        dimension: int,
        noise_mean_param: tuple[int, int] = (-1, 1),
        noise_std_param: tuple[int, int] = (1, 1),
        adj_gen_param: tuple[int, int] = (-2, 2)
    ):
        """Initialize a SEM with a highly sparse adjacency.

        Args:
            dimension: Size of the graph
            noise_mean_param:
                Range of uniform distribution used to generate noise
                distribution means.
            noise_std_param:
                Range of uniform distribution used to generate noise
                distribution standard deviations.
            adj_gen_param:
                Range of uniform distribution used to generate DAG edge
                coefficients. Note that values less than 1.5 in absolute
                value are rounded to zero.
        """
        self.n_var = dimension

        # NOTE: Numerical issues occur if means are not zero.
        self.noise_means = uniform(*noise_mean_param, size=dimension)
        self.noise_stds = uniform(*noise_std_param, size=dimension)

        adj_mat = uniform(*adj_gen_param, size=(dimension, dimension))
        adj_mat = np.tril(adj_mat)

        # Zero out connections near zero
        adj_mat[np.abs(adj_mat) < 1.5] = 0
        np.fill_diagonal(adj_mat, 1)

        self.adj_mat = adj_mat

        super().__init__(self.noise_means, self.noise_stds, self.adj_mat)


class CustomSyntheticDatasetDensity(Dataset):
    """PyTorch Dataset wrapper for Causal SEMs."""

    def __init__(self, X: np.ndarray, device: str = 'cpu'):
        """Initialize torch dataset used to wrap causal SEMs."""
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    def get_dims(self) -> int:
        """Get feature dimensionality of data."""
        return self.data_dim

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.len

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return single datum from dataset."""
        return self.x[index]

    def get_metadata(self) -> dict:
        """Return dataset statistics."""
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }
