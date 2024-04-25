import numpy as np

from data.make_adj_mtx import generate_adj_mat_uniform


def test_make_adj_mtx():
    """Test uniform adjacency generation is lower triangular."""
    dim = 10
    threshold = 0.5

    adj_mat = generate_adj_mat_uniform(dim, threshold)

    assert adj_mat.shape == (dim, dim)
    assert np.sum(np.triu(adj_mat)) == 0
