from __future__ import annotations

import numpy as np

from typing import Callable


class SyntheticMultimodalDataset:
    """Generate a synthetic multimodal dataset.

    If a variable is independent of preceding variables, then it is generated
    as a mixture of Gaussians each with randomly sampled means and variances.

    Otherwise, variables are determined as a weighted non-linear relationship
    of preceding variables plus a standard noise term.
    """

    def __init__(
        self,
        data_dim: int,
        adj_mat_gen_fn: Callable[..., np.ndarray],
        adj_mat_gen_args: dict,
        n_modes: int,
        w_range: tuple[int, int],
        mean_range: tuple[int, int],
        std_range: tuple[int, int],
    ):
        """Initialize generator for a synthetic multimodal dataset.

        Args:
            data_dim: Dimension of generated samples.
            adj_mat_gen_fn: Function which generates adjacency matrix.
            adj_mat_gen_args: Arguments passed to adjacency generator function.
            adj_mat: 2D binary adjacency matrix.
            n_modes: Number of modes in independent variable distributions.
            w_range: Min/max of sampled weights of variable relationships.
            mean_range: Min/max of means of noise distributions.
            std_range: Min/max of standard deviations of noise distributions.
        """
        self.data_dim = data_dim
        self.adj_mat = adj_mat_gen_fn(data_dim, **adj_mat_gen_args)
        self.adj_mat_fn = adj_mat_gen_fn.__name__
        self.adj_mat_args = adj_mat_gen_args

        self.n_modes = n_modes
        self.w_range = w_range
        self.mean_range = mean_range
        self.std_range = std_range

        self.dist_params: None | list[dict] = None

        self.w_mat = self.generate_weight_mat()

    def generate_weight_mat(self) -> np.ndarray:
        """Generate weight matrix describing relationships between nodes.

        Returns:
            2D matrix containing weights of variable relationships.
        """
        w_min, w_max = self.w_range
        weights = np.random.uniform(w_min, w_max, self.adj_mat.shape)
        weight_mat = self.adj_mat * weights
        return weight_mat

    def _generate_independent(
        self,
        n_samples: int,
        params: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Generate samples from a independent variable distribution.

        Args:
            n_samples: Number of data samples to generate.
            params: Optional distributional parameters.
        Returns:
            Array of sampled data of shape (n_samples) and dictionary of
            distributional parameters.
        """
        if params is not None:
            mix_weights = params["mixture"]
            means = params["means"]
            stds = params["stds"]
        else:
            mix_weights = np.random.dirichlet(alpha=np.ones(self.n_modes))
            means = np.random.uniform(*self.mean_range, self.n_modes)
            stds = np.random.uniform(*self.std_range, self.n_modes)

        mode_data = []
        for m in range(self.n_modes):
            # Adds additional sample as a fix for rounding
            mode_samp = int(n_samples * mix_weights[m]) + 1
            mode = np.random.normal(means[m], stds[m], mode_samp)
            mode_data.append(mode)

        row_data = np.concatenate(mode_data)[:n_samples]

        params = {
            "independent": True,
            "mixture": mix_weights,
            "means": means,
            "stds": stds
        }

        return row_data, params

    def _generate_dependent(
        self,
        n_samples: int,
        weights: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Generate samples from a dependent variable distribution.

        Args:
            n_samples: Number of data samples to generate.
            weights: The weights describing dependency on prior variables.
            data: Matrix containing values of prior variables.

        Returns:
            Array of sampled data of shape (n_samples) and dictionary of
            distributional parameters.
        """
        row_mask = weights != 0
        prior_rows_data = data[row_mask]

        nonzero_weights = weights[weights != 0]

        nz_weights = nonzero_weights[:, np.newaxis]
        nz_weights = np.repeat(nz_weights, n_samples, axis=-1)

        new_row_data = np.multiply(nz_weights, prior_rows_data)

        new_row_data = new_row_data ** 2
        new_row_data = np.sum(new_row_data, axis=0)
        new_row_data = new_row_data ** 0.5

        new_row_data += np.random.normal(size=n_samples)

        params = {"independent": False, "weights": weights}
        return new_row_data, params

    def generate_samples(
        self,
        n_samples: int,
        seed: int | None = None
    ) -> np.ndarray:
        """Generate samples from synthetic distribution.

        Initializes distributional parameters if they are not initialized yet.

        Simultaneous initialization / generation is left in as an option as
        this sequence of RNG calls recreates the data distribution from
        the initial submission.

        Args:
            n_samples: Number of data samples to generate.
            seed: Optional fixed random seed.

        Returns:
            Array of sampled data of shape (n_samples, n_features).
        """
        if seed is not None:
            np.random.seed(seed)

        data_arr = np.zeros((self.w_mat.shape[0], n_samples))

        dist_params = []

        for i, row in enumerate(self.w_mat):
            # Check if variable has dependencies
            independent = len(self.w_mat[i][self.w_mat[i] != 0]) == 0

            if self.dist_params is not None:
                row_params = self.dist_params[i]
            else:
                row_params = None

            if independent:
                rd, params = self._generate_independent(n_samples, row_params)
            else:
                rd, params = self._generate_dependent(n_samples, row, data_arr)

            data_arr[i] = rd
            dist_params.append(params)

        if self.dist_params is None:
            self.dist_params = dist_params

        return data_arr.T

    def initialize_distributions(self):
        """Initialize parameters of data generation distributions."""
        dist_params = []

        for row in self.w_mat:
            # Check if variable is independent
            independent = sum(row) == 0

            if independent:
                mixture = np.random.dirichlet(alpha=np.ones(self.n_modes))
                means = np.random.uniform(*self.mean_range, self.n_modes)
                stds = np.random.uniform(*self.std_range, self.n_modes)

                dist_params.append({
                    "independent": True,
                    "mixture": mixture,
                    "means": means,
                    "stds": stds
                })
            else:
                dist_params.append({"independent": False, "weights": row})

        self.dist_params = dist_params

    @classmethod
    def init_from_param(cls, params: list[dict]) -> SyntheticMultimodalDataset:
        """Initialize a generator object from distributional parameters.

        Args:
            params: Dictionary containing data generating distribution
                parameters for independent variables, and the weighting
                of preceding variables for dependent variables.

        Returns:
            Initialized dataset object.
        """
        dataset = cls.__new__(cls)

        dataset.data_dim = len(params)
        w_mat = np.zeros((dataset.data_dim, dataset.data_dim))

        for i, param in enumerate(params):
            if not param["independent"]:
                w_mat[i] = param["weights"]

        dataset.w_mat = w_mat
        dataset.adj_mat = w_mat != 0
        dataset.dist_params = params

        return dataset
