import torch
import wandb
import numpy as np

from strnn.models.strNN import StrNN
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_loop(
    model: StrNN,
    optimizer: Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    train_args: dict
) -> dict | None:
    """
    @param model
    @param optimizer
    @param train_dl: training data loader
    @param val_dl: validation data loader
    @param train_args: dictionary containing other training parameters:
        max_epoch: maximum epoch allowed for training
        patience: max number of epochs without validation loss
            improvement before terminating
    @return: dict of best model state
    """
    best_model_state = None
    best_val = None
    counter = 0

    for epoch in range(1, train_args["max_epoch"]):
        train_losses = []
        val_losses = []

        for batch in train_dl:
            # Training step
            optimizer.zero_grad()
            x = batch
            x_hat, loss = model.get_preds_loss(x)
            train_loss = loss.item()

            loss.backward()
            optimizer.step()

            train_losses.append(train_loss)

        # Validation step
        with torch.no_grad():
            for batch in val_dl:
                x = batch
                x_hat, loss = model.get_preds_loss(x)
                val_loss = loss.item()
                val_losses.append(val_loss)

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)

            # TODO: Add scheduler

            # wandb logging
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss
            })

            if best_val is None or epoch_val_loss < best_val:
                best_val = epoch_val_loss
                best_model_state = model.state_dict()
                counter = 0 # Reset counter since we beat best loss so far
            else:
                counter += 1
                if counter > train_args['patience']:
                    return best_model_state

    return best_model_state