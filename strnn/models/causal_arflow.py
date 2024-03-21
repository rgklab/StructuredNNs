import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Distribution, Laplace, Uniform, Normal
from torch.distributions import TransformedDistribution, SigmoidTransform
from torch.utils.data import DataLoader

from data.causal_sem import CustomSyntheticDatasetDensity

from strnn.models.discrete_flows import AutoregressiveFlow
from strnn.models.discrete_flows import AutoregressiveFlowFactory

from argparse import Namespace


class CausalAutoregressiveFlowWithPrior(nn.Module):
    """Autoregressive flow generated from user specified prior distribution.

    Can be used to define both the Causal StrAF and the baseline CAREFL model,
    which is a Causal StrAF with a fully autoregressive adjacency matrix.

    The CAREFL model developed in Causal Autoregressive Flows,
    by Khemakhem et al. (2021) is available at:
        https://arxiv.org/abs/2011.02268

    The CAREFL and the Causal StrAF can be used to find causal direction
    between pairs of (multivariate) random variables, or to perform
    interventions and answer counterfactual queries.

    Code to perform intervention and counterfactual queries have been adapted
    from CAREFL, but several bugs have been fixed in our implementation. These
    fixes have been noted in the correspond methods.
    """

    def __init__(self, prior: Distribution, flow: AutoregressiveFlow):
        """Initialize an autoregressive flow with a flexible prior.

        Args:
            prior:
                PyTorch distribution describing latent-space prior. Note that
                the prior must match the dimensionality of the latent space,
                as other methods use the prior to generate samples.
            flow:
                AR flow used to perform invertible transformations.
        """
        super().__init__()
        self.prior = prior
        self.flow = flow

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform data-space samples to latent-space and get likelihood.

        Args:
            Data-space samples to normalize of shape (B x D)

        Returns:
            Normalized samples, log prior probability, and Jacobian determinant
        """
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z: torch.Tensor) -> torch.Tensor:
        """Transform latent-space samples to data-space using flow.

        Args:
            z: Samples in latent-space of shape (B x D)

        Returns:
            Samples in data-space after flow transformation of shape (B x D)
        """
        xs = self.flow.invert(z)
        return xs

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate data-space samples using inverse flow transformation.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Samples in data space generated from flow distribution
        """
        z = self.prior.sample(torch.Size((num_samples,)))
        xs = self.flow.invert(z)
        return xs

    def log_likelihood(self, x: torch.Tensor) -> np.ndarray:
        """Compute log likelihood of samples in data-space.

        Args:
            x: Data sample of shape (B x D)

        Returns:
            Log likelihood of data samples
        """
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det).detach().cpu().numpy()

    def predict_intervention(
        self,
        x0_val: float,
        n_samples: int = 100,
        iidx: int = 0
    ) -> np.ndarray:
        """Predict the value of x given an intervention do(x_iidx = x0_val).

        As explained in Appendix E.1 of the CAREFL paper, we do not need to
        invert the flow to perform interventions, but doing so provides a
        simpler and more efficient algorithm. Thus, this implementation will
        follow Algorithm 2 of Appendix E.1.

        However, the CAREFL implementation is actually incorrect, and has
        been fixed here. The intervention proceeds as below:

        1) sample noises z from base distribution
        2) pass z through flow to get initial samples for x
        3) invert flow to find corresponding entry for z_iidx at x_iidx=x0_val
        4) pass z again through flow to get samples for x

        Args:
            x0_val: Interventional value
            n_samples: Number of samples used to estimate interventional effect
            iidx: Index of variable to intervene on

        Return:
            Estimate of interventional effect
        """
        # Sample from prior and ensure z_intervention_index = z_int
        z = self.prior.sample(torch.Size((n_samples,)))

        # Pass z through flow to get initial samples for x
        x_init = self.backward(z)

        # Invert flow to infer value of latent corresponding to
        # interventional variable
        x_init[:, iidx] = x0_val
        z_int = self.forward(x_init)[0][:, iidx]
        z[:, iidx] = z_int

        # Propagate the latent sample through flow
        x = self.backward(z)

        # Sanity check: check x_intervention_index == x0_val
        assert (torch.abs(x[:, iidx] - x0_val) < 1e-3).all()

        int_est = x.mean(0).reshape((1, z.shape[1]))
        return int_est.detach().cpu().numpy()

    def predict_counterfactual(
        self,
        x_obs: torch.Tensor | np.ndarray,
        cf_val: float,
        iidx: int = 0
    ) -> np.ndarray:
        """Predict counterfactual of setting dimension iidx of x_obs to cf_val.

        For clarity, we restrict counterfacutal queries to a single example.
        However, this method can easily be extended to accept an array of
        counterfactual values instead.

        Given observation x_obs we estimate the counterfactual of setting
        x_obs[intervention_index] = cf_val

        This proceeds in 3 steps:
         1) abduction - pass-forward through flow to infer latents for x_obs
         2) action - pass-forward again for latent associated with cf_val
         3) prediction - backward pass through the flow

        Args:
            x_obs: Observed data sample to perform counterfactual query.
            cf_val: Counterfactual value.
            iidx: Feature dimension of counterfactual value.

        Returns:
            Counterfactual of observed data sample
        """
        if x_obs.shape[0] != 1:
            raise ValueError("Counterfactual only accepts single example.")

        if isinstance(x_obs, np.ndarray):
            x_obs = torch.from_numpy(x_obs)

        # abduction:
        z_obs = self.forward(x_obs)[0]
        # action (get latent variable value under counterfactual)
        x_cf = torch.clone(x_obs)
        x_cf[0, iidx] = cf_val
        z_cf_val = self.forward(x_cf)[0][0, iidx]
        z_obs[0, iidx] = z_cf_val

        # prediction (pass through the flow):
        x_post_cf = self.backward(z_obs)
        return x_post_cf.detach().cpu().numpy()


class CausalARFlowTrainer:
    """Wrapper around the Causal StrAF / CAREFL to allow easier training."""

    def __init__(self, config: Namespace):
        """Initialize a causal flow trainer.

        Args:
            config: Dictionary of model arguments. See below for parameters

            config.arch_config: Config passed to AutoregressiveFlow init
            config.device: CUDA device used for training
            config.flow.prior_dist: Prior distribution used for likelihood
            config.optim.adam_beta: Adam optimizer beta_1 coefficient
            config.optim.amsgrad: Whether to use amsgrad in Adam optimizer
            config.optim.lr: Optimization learning rate
            config.optim.scheduler: Whether a learning rate scheduler is used
            config.optim.weight_decay: Adam weight decay value
            config.training.batch_size: Batch size used in training
            config.training.early_stop_patience: Epochs before early stopping
            config.training.epoch: Number of training epochs
            config.training.seed: Random seed used for training
            config.training.split: Training split ratio
            config.training.verbose: Whether training logs are printed
        """
        self.config: Namespace = config
        self.arch_config: dict = config.arch_config
        self.prior_dist: Distribution = config.flow.prior_dist

        self.device: torch.device = config.device

        self.adam_beta: float = config.optim.beta1
        self.amsgrad: bool = config.optim.amsgrad
        self.lr: float = config.optim.lr
        self.scheduler: bool = config.optim.scheduler
        self.weight_decay: float = config.optim.weight_decay

        self.batch_size: int = config.training.batch_size
        self.epochs: int = config.training.epochs
        self.seed: int = config.training.seed
        self.split_ratio: float = config.training.split
        self.verbose: bool = config.training.verbose
        self.stopping_patience: int = config.training.early_stop_patience

        self.dim: int | None = None
        self.flow: CausalAutoregressiveFlowWithPrior | None = None

    def fit_to_sem(self, data: np.ndarray) -> tuple[list[float], list[float]]:
        """Fits SEM assuming data columns follow the causal ordering.

        Args:
            data:
                A dataset whose columns are ordered following the causal
                ordering $pi$. The causal ordering can be a result of causal
                discovery algorithm, or expert judgement.
        """
        train_dl, val_dl = self._get_datasets(data)

        torch.manual_seed(self.seed)
        self.flow = self._get_flow_arch()

        train_losses, val_losses = self._train(self.flow, train_dl, val_dl)

        return train_losses, val_losses

    def _get_datasets(self, data: np.ndarray) -> tuple[DataLoader, DataLoader]:
        """Get train/val dataset splits from overall design matrix.

        Args:
            dataset: Observed samples

        Returns:
            Training and validation dataloaders
        """
        assert isinstance(data, np.ndarray)

        self.dim = data.shape[-1]

        train_np, val_np = train_test_split(data, train_size=self.split_ratio)

        train = CustomSyntheticDatasetDensity(train_np.astype(np.float32))
        val = CustomSyntheticDatasetDensity(val_np.astype(np.float32))

        train_dl = DataLoader(train, shuffle=True, batch_size=self.batch_size)
        val_dl = DataLoader(val, shuffle=False, batch_size=self.batch_size)

        return train_dl, val_dl

    def _get_flow_arch(self) -> CausalAutoregressiveFlowWithPrior:
        """Return a normalizing flow according to the config file.

        Returns:
            Initialized Causal Autoregressive Flow (Causal StrAF or CAREFL).
        """
        if self.dim is None:
            raise RuntimeError("self._get_datasets must be called first.")

        prior: Distribution | None = None

        # Setup priors
        if self.prior_dist == 'laplace':
            prior = Laplace(
                torch.zeros(self.dim).to(self.device),
                torch.ones(self.dim).to(self.device)
            )
        elif self.prior_dist == 'normal':
            prior = Normal(
                torch.zeros(self.dim).to(self.device),
                torch.ones(self.dim).to(self.device)
            )
        else:
            prior = TransformedDistribution(
                Uniform(
                    torch.zeros(self.dim).to(self.device),
                    torch.ones(self.dim).to(self.device)
                ),
                SigmoidTransform().inv
            )

        # Build flow model
        model_factory = AutoregressiveFlowFactory(self.arch_config)
        flow = model_factory.build_flow()

        assert isinstance(flow, AutoregressiveFlow)

        return CausalAutoregressiveFlowWithPrior(prior, flow).to(self.device)

    def _get_optimizer(
        self,
        model: CausalAutoregressiveFlowWithPrior,
    ) -> tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau | None]:
        """Return an optimizer according to the config file.

        Args:
            model: Model to optimize

        Returns:
            Initialized Adam optimizer, and optionally a LR scheduler
        """
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta, 0.999),
            amsgrad=self.amsgrad
        )

        if self.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=5,
                verbose=self.verbose
            )
        else:
            scheduler = None

        return optimizer, scheduler

    def _train(
        self,
        flow: CausalAutoregressiveFlowWithPrior,
        train_dl: DataLoader,
        val_dl: DataLoader
    ) -> tuple[list, list]:
        """Train causal autoregressive flow.

        Implements early stopping if the stopping_patience attribute is not
        zero. This halts training if validation loss does not improve the
        number of epochs specified by stopping_patience. The model is then set
        to the weights which yielded lowest loss.

        Args:
            flow: Causal autoregressive flow to train
            train_dl: Train dataloader
            val_dl: Validation dataloader

        Returns:
            Train and validation losses throughout training.
        """
        optimizer, scheduler = self._get_optimizer(flow)

        flow.train()

        train_losses = []
        val_losses = []

        best_val_loss = None
        best_weights = flow.state_dict()
        patience_counter = 0

        for e in range(self.epochs):
            epoch_t_loss = []
            epoch_v_loss = []

            for train_batch in train_dl:
                optimizer.zero_grad()
                train_batch = train_batch.to(self.device)

                # compute loss
                _, prior_logprob, log_det = flow(train_batch)
                train_loss = -torch.sum(prior_logprob + log_det)
                epoch_t_loss.append(train_loss.item())

                # optimize
                train_loss.backward()
                optimizer.step()

            with torch.no_grad():
                flow.eval()

                for val_batch in val_dl:
                    val_batch = val_batch.to(self.device)

                    # compute loss
                    _, prior_logprob, log_det = flow(val_batch)
                    val_loss = -torch.sum(prior_logprob + log_det)
                    epoch_v_loss.append(val_loss.item())

                flow.train()

            epoch_t_mean = np.mean(epoch_t_loss)
            epoch_v_mean = np.mean(epoch_v_loss)

            # Handle early stopping
            if self.stopping_patience > 0:
                if best_val_loss is None or epoch_v_mean < best_val_loss:
                    best_weights = flow.state_dict()
                    best_val_loss = epoch_v_mean
                    patience_counter = 0
                elif patience_counter > self.stopping_patience:
                    if self.verbose:
                        print("Early stopping training.")
                    break
                else:
                    patience_counter += 1

            if self.scheduler and scheduler is not None:
                scheduler.step(epoch_t_mean)

            if self.verbose and e % 10 == 0:
                msg = "Epoch {}/{} \tTrain loss: {}\t Val loss: {}"
                print(msg.format(e, self.epochs, epoch_t_mean, epoch_v_mean))

            train_losses.append(epoch_t_mean)
            val_losses.append(epoch_v_mean)

        # Update model weights to best
        if self.stopping_patience > 0:
            flow.load_state_dict(best_weights)

        return train_losses, val_losses
