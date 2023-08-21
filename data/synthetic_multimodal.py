from __future__ import annotations

import numpy as np

from typing import Callable


class SyntheticMultimodalDataset:
    """
    Generates a synthetic multimodal dataset.

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
        """Initializes generator for a synthetic multimodal dataset.

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

        self.w_mat = self.generate_weight_mat()
        self.dist_params = self.generate_distributions()

    def generate_weight_mat(self) -> np.ndarray:
        """Generates weight matrix describing relationships between nodes.

        Returns:
            2D matrix containing weights of variable relationships.
        """
        w_min, w_max = self.w_range
        weights = np.random.uniform(w_min, w_max, self.adj_mat.shape)
        weight_mat = self.adj_mat * weights
        return weight_mat

    def generate_distributions(self) -> list[dict]:
        """Generates parameters of data generation distributions."""
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

        return dist_params

    def generate_samples(self, n_samples: int, seed: int = None) -> np.ndarray:
        """Generates samples from data distribution.

        Args:
            n_samples: Number of data samples to generate.
            seed: Optional random seed to use in data sampling.
        Returns:
            Array of sampled data of shape (n_samples, n_features).
        """
        if seed is not None:
            np.random.seed(seed)

        data_arr = np.zeros((self.w_mat.shape[0], n_samples))

        for i, params in enumerate(self.dist_params):
            if params["independent"]:
                mode_data = []
                for m in range(self.n_modes):
                    # Adds additional sample as a fix for rounding
                    mode_samples = int(n_samples * params["mixture"][m]) + 1
                    mode = np.random.normal(params["means"][m],
                                            params["stds"][m],
                                            mode_samples)
                    mode_data.append(mode)

                row_data = np.concatenate(mode_data)[:n_samples]
                data_arr[i] = row_data
            else:
                row_mask = params["weights"] != 0
                prior_rows_data = data_arr[row_mask]

                nonzero_weights = params["weights"][params["weights"] != 0]

                nz_weights = nonzero_weights[:, np.newaxis]
                nz_weights = np.repeat(nz_weights, n_samples, axis=-1)

                new_row_data = np.multiply(nz_weights, prior_rows_data)

                new_row_data = new_row_data ** 2
                new_row_data = np.sum(new_row_data, axis=0)
                new_row_data = new_row_data ** 0.5

                new_row_data += np.random.normal(size=n_samples)

                data_arr[i] = new_row_data

        return data_arr.T

    @classmethod
    def init_from_param(cls, params: list[dict]) -> SyntheticMultimodalDataset:
        """Initializes a generator object from distributional parameters.

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
