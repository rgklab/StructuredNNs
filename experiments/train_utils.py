import datetime
import pathlib
import os

import numpy as np

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb

from strnn.models.continuous_flows import ContinuousFlowFactory
from strnn.models.discrete_flows import AutoregressiveFlowFactory

from strnn.models.normalizing_flow import NormalizingFlow


MODEL_CONSTR_MAP = {
    "CNF": ContinuousFlowFactory,
    "ANF": AutoregressiveFlowFactory,
}


def train_loop(
    model: NormalizingFlow,
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    train_dl: DataLoader,
    val_dl: DataLoader,
    train_args: dict
) -> dict | None:
    """Training loop for normalizing flows, including validation step.

    Args:
        model: Normalizing flow model to train.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch learning rate scheduler. Can be None.
        train_dl: Training set dataloader.
        val_dl: Validation set dataloader.
        train_args: Dictionary containing other training parameters.
    Returns:
        Dictionary of the best model state.
    """
    best_model_state = None
    best_val = None
    counter = 0

    for epoch in range(1, train_args["max_epoch"]):
        train_losses = []
        for batch in train_dl:
            optimizer.zero_grad()
            z, jac = model(batch)
            loss = model.compute_loss(z, jac)
            train_loss = loss.item()

            loss.backward()
            optimizer.step()

            train_losses.append(train_loss)

        with torch.no_grad():
            val_losses = []
            for batch in val_dl:
                z, jac = model(batch)
                val_loss = model.compute_loss(z, jac)
                val_losses.append(val_loss.item())

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                elif isinstance(scheduler, LRScheduler):
                    scheduler.step()
                else:
                    raise RuntimeError("Unexpected scheduler type.")

            if train_args["use_wandb"]:
                wandb.log({"train_loss": epoch_train_loss,
                           "val_loss": epoch_val_loss})

            if best_val is None or epoch_val_loss < best_val:
                best_val = epoch_val_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter > train_args["patience"]:
                    return best_model_state

    return best_model_state


def check_wandb_run(config: dict, wandb_name: str):
    """Verify if wandb run with same config already is completed.

    TODO: Expand function to cover more config keys.

    Args:
        config: Dictionary of run configuration values.
        wandb_name: Name of current wandb project.

    Returns:
        Whether a run with current config is finished or running.
    """
    api = wandb.Api()
    runs = api.runs(
        wandb_name,
        filters={
            "config.conditioner_type": config["conditioner_type"],
            "config.data_random_seed": config["data_random_seed"],
            "config.flow_steps": config["flow_steps"],
            "config.lr": config["lr"],
            "config.model_config": config["model_config"],
            "config.n_param_per_var": config["n_param_per_var"],
            "config.conditioner_hid_dim": config["conditioner_hid_dim"],
            "config.umnn_int_hid_dim": config["umnn_int_hid_dim"],
        })

    if len(runs) == 1:
        return runs[0].state in ["finished", "running"]
    elif len(runs) > 1:
        raise RuntimeError("Unexpected multiple wandb runs.")
    return False


def persist_wandb(
    wandb_run: wandb.apis.public.Run,
    model_state: dict,
    temp_path: str = "./wandb"
):
    """Persist model state to WandB, and deletes temp artifact.

    Args:
        wandb_run: WandB run to log artifact to.
        model_state: Dictionary of model weights.
        temp_path: Path to temporary directory used to store weights prior to
            upload onto wandb.
    """
    model_dir = pathlib.Path("./wandb")
    model_dir.mkdir(exist_ok=True)

    timestamp = str(datetime.datetime.now())[:19]
    model_path = "{}/{}.pt".format(model_dir, timestamp)
    torch.save(model_state, model_path)

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)

    wandb_run.log_artifact(artifact)

    os.remove(model_path)
