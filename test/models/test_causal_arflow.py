import pytest
from unittest.mock import Mock

import numpy as np

import torch
import torch.distributions as dist

from strnn.models.causal_arflow import CausalAutoregressiveFlowWithPrior
from strnn.models.discrete_flows import AutoregressiveFlow


@pytest.fixture
def mock_flow():
    """Mock autoregressive flow which has identity forward and backwards."""
    def identity_forward(in_val):
        return in_val, in_val

    def identity_back(in_val):
        return in_val

    mock_flow = Mock(spec=AutoregressiveFlow)

    mock_flow.forward = Mock(side_effect=identity_forward)
    mock_flow.invert = Mock(side_effect=identity_back)

    return mock_flow


def test_forward(mock_flow):
    """Test that the CausalAF simply passthroughs the AF forward."""
    test_in = torch.randn(8, 16)

    prior_dist = dist.Normal(0, 1)
    model = CausalAutoregressiveFlowWithPrior(prior_dist, mock_flow)

    z, prior_prob, _ = model.forward(test_in)

    assert torch.all(z == test_in)
    assert torch.all(prior_prob == prior_dist.log_prob(test_in).sum(1))


def test_backward(mock_flow):
    """Test that the CausalAF simply passthroughs the AF backward."""
    test_in = torch.randn(8, 16)

    prior = dist.Normal(0, 1)
    model = CausalAutoregressiveFlowWithPrior(prior, mock_flow)

    x = model.backward(test_in)

    assert torch.all(x == test_in)


def test_sample(mock_flow):
    """Test Casual AF sampling."""
    prior = dist.Normal(torch.zeros(3), torch.ones(3))
    model = CausalAutoregressiveFlowWithPrior(prior, mock_flow)

    samples = model.sample(100)

    assert samples.shape == (100, 3)


def test_interventional(mock_flow):
    """Test interventional estimation.

    TODO: This could likely be made a more robust test by creating a mock
    flow which actually depends on the value of the intervention, but alas...
    """
    n_samp = 10
    n_dim = 3

    # Mock a fixed prior distribution sample
    mock_prior = Mock(spec=dist.Normal)
    mock_prior.sample.return_value = torch.ones(n_samp, n_dim)

    model = CausalAutoregressiveFlowWithPrior(mock_prior, mock_flow)

    int_est = model.predict_intervention(2, n_samp, 1)

    assert int_est.shape == (1, n_dim)
    assert np.all(int_est == np.array([[1, 2, 1]]))


def test_counterfactual_multi_errors(mock_flow):
    """Test that feeding multiple samples into counterfactual raises error."""
    with pytest.raises(ValueError):
        test_in = torch.randn(10, 3)

        prior = dist.Normal(0, 1)
        model = CausalAutoregressiveFlowWithPrior(prior, mock_flow)

        model.predict_counterfactual(test_in, 3, 1)


def test_counterfactual(mock_flow):
    """Test that counterfactual inference works.

    TODO: This could likely be made a more robust test by creating a mock
    flow which actually returns a conditional output based on correct
    substitution of counterfactual.
    """
    prior = dist.Normal(0, 1)
    model = CausalAutoregressiveFlowWithPrior(prior, mock_flow)

    test_in = torch.ones(1, 3)
    ctf_est = model.predict_counterfactual(test_in, 3, 1)

    assert ctf_est.shape == (1, 3)
    assert np.all(ctf_est == np.array([[1, 3, 1]]))
