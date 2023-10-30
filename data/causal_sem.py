import numpy as np
import torch
from numpy.random import normal, uniform, laplace
from torch.utils.data import Dataset


class LinAddSEM:
    """
    Defines a linear additive SEM and sampling operations.
    """
    def __init__(self, noise_mean, noise_stds, adj_mat, noise_dist=normal):
        """Initialize SEM.

        All input variables should be np.array.

        Assumes autoregressive causal ordering, meaning adjacency matrix must
        be lower triangular.
        """
        # Check that SEM specification is valid
        assert(len(noise_mean) == len(noise_stds))

        assert(len(adj_mat.shape) == 2)
        assert(adj_mat.shape[0] == adj_mat.shape[1])
        assert(len(noise_mean) == adj_mat.shape[0])
        assert(np.allclose(adj_mat, np.tril(adj_mat)))

        self.n_var = len(noise_mean)

        self.noise_mean = noise_mean
        self.noise_stds = noise_stds
        self.noise_dist = noise_dist
        self.adj_mat = adj_mat

    def generate_sample(self):
        """Generates a sample from specified SEM."""
        e = self.noise_dist(self.noise_mean, self.noise_stds)

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat

    def generate_samples(self, n_samp):
        return np.array([self.generate_sample() for _ in range(n_samp)])

    def generate_intervention(self, int_val):
        e = self.noise_dist(self.noise_mean, self.noise_stds)

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            if int_val[i] is not None:
                out_mat[i] = int_val[i]
            else:
                out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat

    def generate_int_dist(self, int_val, n_samp, return_mean=True):
        samples = []
        for _ in range(n_samp):
            samples.append(self.generate_intervention(int_val))

        if return_mean:
            return np.mean(np.array(samples), axis=0)
        else:
            return np.array(samples)

    def generate_ctf_obs(self):
        """Generates an obs from specified SEM, return both obs and noise."""
        e = self.noise_dist(self.noise_mean, self.noise_stds)

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat, e

    def generate_counterfactual(self, e, ctf_val):

        out_mat = np.zeros_like(e)

        for i, row in enumerate(self.adj_mat):
            if ctf_val[i] is not None:
                out_mat[i] = ctf_val[i]
            else:
                out_mat[i] = out_mat.dot(row) + row[i] * e[i]

        return out_mat

    def get_carefl_ds(self, n_samp):
        X = self.generate_samples(n_samp)

        # Generate binary adjacency matrix
        cfl_adj_mat = (self.adj_mat != 0).astype(int)
        np.fill_diagonal(cfl_adj_mat, 0)

        return X, None, cfl_adj_mat


class RandomSEM:
    """Initializes a random LinAddSEM."""

    def __init__(self, dimension, noise_mean_param=(-2, 2),
                 noise_std_param=(1, 10), adj_gen_param=(-2, 2)):
        """Initialize SEM.

        Args:
            dimension (int): Size of the graph.
            noise_mean_param (float, float): Parameters to generate noise mean.
            noise_std_param (float, float): Parameters to generate noise std.
            adj_gen_param (float, float): Parameters to generate adjacency
                                          weight matrix.
        """
        self.n_var = dimension

        # NOTE: Numerical issues occur if means are not zero.
        self.noise_means = uniform(*noise_mean_param, size=dimension)
        self.noise_stds = uniform(*noise_std_param, size=dimension)

        adj_mat = uniform(*adj_gen_param, size=(dimension, dimension))
        self.adj_mat = np.tril(adj_mat)

        self.sem = LinAddSEM(self.noise_means, self.noise_stds, self.adj_mat)

    def generate_samples(self, n_samples):
        return self.sem.generate_samples(n_samples)

    def generate_int_dist(self, int_val, n_samples):
        return self.sem.generate_int_dist(int_val, n_samples)

    def get_adj_mat(self):
        bin_adj_mat = (self.adj_mat != 0).astype(int)
        np.fill_diagonal(bin_adj_mat, 0)
        return bin_adj_mat


class SparseSEM:
    """Initializes a LinAddSEM with many independencies."""

    def __init__(self, dimension, noise_mean_param=(-1, 1),
                 noise_std_param=(1, 1), adj_gen_param=(-2, 2)):
        """Initialize SEM.

        Args:
            dimension (int): Size of the graph.
            noise_mean_param (float, float): Parameters to generate noise mean.
            noise_std_param (float, float): Parameters to generate noise std.
            adj_gen_param (float, float): Parameters to generate adjacency
                                          weight matrix.
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

        self.sem = LinAddSEM(self.noise_means, self.noise_stds, self.adj_mat)

    def generate_samples(self, n_samples):
        return self.sem.generate_samples(n_samples)

    def generate_int_dist(self, int_val, n_samples, return_mean=True):
        return self.sem.generate_int_dist(int_val, n_samples, return_mean)

    def generate_ctf_obs(self):
        return self.sem.generate_ctf_obs()

    def generate_counterfactual(self, e, ctf_val):
        return self.sem.generate_counterfactual(e, ctf_val)

    def get_adj_mat(self):
        bin_adj_mat = (self.adj_mat != 0).astype(int)
        np.fill_diagonal(bin_adj_mat, 0)
        return bin_adj_mat
    

class CustomSyntheticDatasetDensity(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }