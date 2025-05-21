"""Multilayer Regression

Original version used the pytorch_lightning from Nemo.
This version uses the lightning module.
"""

import json
import logging
import os
import tarfile
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from jaxtyping import Float
from lightning.pytorch import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


class MLP(torch.nn.Module):
    """MLP layer.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        activation: Activation function to use.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "tanh"
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)

        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ResidualMLP(torch.nn.Module):
    """Residual MLP module.

    Args:
        dim: Feature dimension.
        num_layers: Number of MLP layers.
        hidden_dim: Hidden dimension for each MLP layer.
        activation: Activation function to use.
    """

    def __init__(
        self, dim: int, num_layers: int, hidden_dim: int, activation: str = "tanh"
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [MLP(dim, hidden_dim, dim, activation) for _ in range(num_layers)]
        )

    def forward(self, x):
        input_x = x
        for layer in self.layers:
            x = layer(x)
            x = x + input_x  # Residual connection
            input_x = x
        return x


class MultiBlockRegressor(torch.nn.Module):
    """A multi-block regressor (core architecture).

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension for each block.
        output_dim: Output dimension.
        num_blocks: Number of residual blocks.
        num_layers_per_block: Number of MLP layers per block.
        activation: Activation function to use.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 1,
        num_blocks: int = 2,
        num_layers_per_block: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Create the blocks
        self.blocks = nn.ModuleList()

        # First block takes the input dimension
        first_block = ResidualMLP(
            dim=input_dim,
            num_layers=num_layers_per_block,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        self.blocks.append(first_block)
        setattr(self, f"block_0", first_block)

        # Subsequent blocks maintain the same dimension
        for i in range(1, num_blocks):
            block = ResidualMLP(
                dim=input_dim,
                num_layers=num_layers_per_block,
                hidden_dim=hidden_dim,
                activation=activation,
            )
            self.blocks.append(block)
            setattr(self, f"block_{i}", block)

        # Final output projection if needed
        if input_dim != output_dim:
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        # Pass through each block
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.output_proj(x)

        return x


class MultiBlockRegressorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._validation_dl

    def test_dataloader(self):
        return self._test_dl


# Inherit directly from pl.LightningModule and implement methods needed
class MultiBlockRegressorPT(pl.LightningModule):
    """A multi-block regressor wrapped as a Lightning Module.

    Uses composition to contain a MultiBlockRegressor instance.
    """

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # First call LightningModule's init
        super().__init__()

        # Store the config received. This config is expected to contain
        # 'arch', 'optim', and dataset keys at its root level
        self.cfg = cfg

        # Now create the regressor as a member variable
        # Access architectural parameters from cfg.arch
        self.regressor = MultiBlockRegressor(
            input_dim=cfg.arch.input_dim,
            hidden_dim=cfg.arch.hidden_dim,
            output_dim=cfg.arch.output_dim,
            num_blocks=cfg.arch.num_blocks,
            num_layers_per_block=cfg.arch.num_layers_per_block,
            activation=cfg.arch.activation,
        )

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Create dataloaders
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate to the regressor's forward method
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_cfg = self.cfg.optim
        optimizer = torch.optim.Adam(
            self.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay
        )

        if (
            hasattr(optim_cfg, "sched")
            and OmegaConf.select(optim_cfg, "sched.name") is not None
        ):
            scheduler_cfg = optim_cfg.sched
            if scheduler_cfg.name == "CosineAnnealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=scheduler_cfg.T_max, eta_min=scheduler_cfg.min_lr
                )
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            else:
                raise NotImplementedError(
                    f"Scheduler {scheduler_cfg.name} not implemented"
                )
        else:
            return optimizer

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )
        logging.info("Train DataLoader prepared")

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )
        logging.info("Validation DataLoader prepared")

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = self._get_dataloader_from_config(
            test_data_config, shuffle=False
        )
        logging.info("Test DataLoader prepared")

    def _get_dataloader_from_config(
        self, config: Union[DictConfig, Dict], shuffle: bool = False
    ):
        if config is None:
            return None

        try:
            file_path = config.file_path
            batch_size = config.batch_size
            num_workers = config.get("num_workers", 0)
            pin_memory = config.get("pin_memory", False)

            print(f"Loading data from: {file_path}")
            data = np.load(file_path)
            x = data["x"]
            y = data["y"]

            dataset = torch.utils.data.TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            print(f"DataLoader created successfully for {file_path}.")
            return dataloader
        except Exception as e:
            print(f"Error creating dataloader from config: {e}")
            raise

    # Add methods needed for PyTorch Lightning
    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._validation_dl

    def test_dataloader(self):
        return self._test_dl

    # Simple save and restore methods to mimic ModelPT
    def save_to(self, save_path):
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")


class LossHistory(pl.Callback):
    """Callback to store training and validation losses for plotting."""

    def __init__(self):
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Note: trainer.callback_metrics contains metrics logged with on_epoch=True
        # The keys are the names used in self.log().
        # If you log "train_loss" with on_epoch=True, it appears here as "train_loss_epoch".
        # If you log "train_loss" without _epoch suffix, it might be available directly
        # depending on PL version and exact logging setup.
        # Let's check both just in case, but typically it's metric_name_epoch.
        train_loss = trainer.callback_metrics.get("train_loss_epoch")
        if train_loss is None:
            # Fallback to the name used in self.log if _epoch isn't appended
            train_loss = trainer.callback_metrics.get("train_loss")

        if train_loss is not None:
            # Ensure we handle both cases where train_loss could be a tensor or a Python number
            if isinstance(train_loss, torch.Tensor):
                self.train_losses.append(train_loss.cpu().item())
            else:
                self.train_losses.append(float(train_loss))

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Note: trainer.callback_metrics contains metrics logged with on_epoch=True
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        if val_loss is None:
            # Fallback to the name used in self.log if _epoch isn'loss appe
            val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is not None:
            # Ensure we handle both cases
            if isinstance(val_loss, torch.Tensor):
                self.val_losses.append(val_loss.cpu().item())
            else:
                self.val_losses.append(float(val_loss))


class PredictionPlotter(pl.Callback):
    """Callback to plot model predictions on test data periodically."""

    def __init__(
        self,
        test_data_path: str,
        loss_history_callback: LossHistory,
        plot_interval: int = 2,
    ):
        super().__init__()
        self.test_data_path = test_data_path
        self.loss_history = loss_history_callback  # Store the LossHistory instance
        self.plot_interval = plot_interval
        # Load test data once in init
        try:
            data = np.load(test_data_path)
            self.x_test_plot = data["x"]
            self.y_test_plot = data["y"]
            logging.info(f"Loaded test data for plotting from {test_data_path}")
        except Exception as e:
            logging.error(f"Error loading test data for plotting: {e}")
            self.x_test_plot = None
            self.y_test_plot = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Plot every plot_interval epochs, and optionally for the first few epochs
        current_epoch = trainer.current_epoch
        # Plot if it's a multiple of plot_interval (after the first epoch to ensure val_loss is available)
        # or for epoch 0 if you want to see initial state
        if (
            current_epoch > 0 and (current_epoch + 1) % self.plot_interval == 0
        ) or current_epoch == 0:
            if (
                self.x_test_plot is not None
                and self.loss_history.train_losses
                and self.loss_history.val_losses
            ):
                print(f"\nPlotting predictions for epoch {current_epoch + 1}...")

                # Get the most recent losses from the LossHistory callback
                # Since on_validation_epoch_end is called after validation for the epoch,
                # the latest loss should be the last element in the lists.
                try:
                    train_loss_val = self.loss_history.train_losses[-1]
                    val_loss_val = self.loss_history.val_losses[-1]
                except IndexError:
                    # Handle cases where lists might still be empty in very early stages
                    train_loss_val = float("nan")
                    val_loss_val = float("nan")
                    print(
                        f"Warning: Loss history lists are empty at epoch {current_epoch + 1}. Cannot include losses in plot title."
                    )

                # Get predictions
                pl_module.eval()
                with torch.no_grad():
                    # Ensure tensor is on the correct device
                    x_test_tensor_plot = torch.tensor(
                        self.x_test_plot, dtype=torch.float32
                    ).to(pl_module.device)
                    y_pred_plot = pl_module(x_test_tensor_plot).cpu().numpy()
                pl_module.train()  # Set back to training mode

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(
                    self.x_test_plot,
                    self.y_test_plot,
                    "bo",
                    markersize=3,
                    alpha=0.5,
                    label="Original Data (Target)",
                )
                plt.plot(self.x_test_plot, y_pred_plot, "r-", label="Model Prediction")
                plt.xlabel("x")
                plt.ylabel("y")

                # Format title with epoch and losses
                title = f"Epoch {current_epoch + 1} Predictions | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss_val:.4f}"
                plt.title(title)

                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save plot with epoch number in filename
                plot_filename = f"prediction_epoch_{current_epoch + 1}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"Prediction plot saved as '{plot_filename}'.")
            elif self.x_test_plot is None:
                print(
                    f"Skipping plot for epoch {current_epoch + 1}: Test data not loaded."
                )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    # Create the model
    config = OmegaConf.load("config/multiblock_config.yaml")

    # Ensure config has 'trainer' and 'data' sections at the root,
    # and that config.model has 'arch', 'optim', and dataset keys.
    # Assuming the YAML structure is now:
    # trainer: { max_epochs: ... }
    # data: { train_ds: { ... }, validation_ds: { ... }, test_ds: { ... } }
    # model: { arch: { ... }, optim: { ... } }

    # If config structure is trainer:{..}, data:{..}, model:{..}
    # model_init_cfg should contain model.arch, model.optim, data.train_ds, data.val_ds, data.test_ds
    # Need to create a merged config for the model constructor
    model_init_cfg = OmegaConf.create()
    # Check if 'model' and 'arch' exist before accessing
    if "model" in config and "arch" in config.model:
        model_init_cfg.arch = config.model.arch
    else:
        raise ValueError("Config must contain 'model.arch' for model initialization.")

    if "model" in config and "optim" in config.model:
        model_init_cfg.optim = config.model.optim
    else:
        # Handle case where optim might be missing if it's optional
        print("Warning: 'model.optim' not found in config.")
        model_init_cfg.optim = OmegaConf.create({})  # Provide empty dict config

    # Check if 'data' and its sub-keys exist
    if "data" in config and "train_ds" in config.data:
        model_init_cfg.train_ds = config.data.train_ds
    else:
        raise ValueError("Config must contain 'data.train_ds' for data setup.")

    if "data" in config and "validation_ds" in config.data:
        model_init_cfg.validation_ds = config.data.validation_ds
    else:
        print(
            "Warning: 'data.validation_ds' not found in config. Validation might be skipped."
        )
        model_init_cfg.validation_ds = None  # Set to None if missing

    if "data" in config and "test_ds" in config.data:
        model_init_cfg.test_ds = config.data.test_ds
    else:
        print("Warning: 'data.test_ds' not found in config. Test might be skipped.")
        model_init_cfg.test_ds = None  # Set to None if missing

    model = MultiBlockRegressorPT(cfg=model_init_cfg)

    # Setup loss history callback FIRST
    loss_history = LossHistory()

    # Setup prediction plotter callback
    # Pass the file path to the test data config AND the loss_history instance
    prediction_plotter = PredictionPlotter(
        test_data_path=config.data.test_ds.file_path,
        loss_history_callback=loss_history,  # Pass the LossHistory instance
        plot_interval=config.trainer.plot_interval,
    )

    # Create trainer
    # Check if 'trainer' and 'max_epochs' exist
    max_epochs = 10  # Default
    if "trainer" in config and "max_epochs" in config.trainer:
        max_epochs = config.trainer.max_epochs
    else:
        print(
            f"Warning: 'trainer.max_epochs' not found in config. Using default max_epochs={max_epochs}."
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="cpu",  # Ensure this is set correctly for your hardware
        devices=1,
        callbacks=[loss_history, prediction_plotter],  # Add both callbacks here
    )

    # Train the model
    start = time.time()
    # Data setup methods must be called before trainer.fit
    # These methods use the dataset configs stored in model.cfg (which is model_init_cfg)
    model.setup_training_data(model.cfg.train_ds)
    # Only setup validation if the config exists
    if model.cfg.validation_ds is not None:
        model.setup_validation_data(model.cfg.validation_ds)

    trainer.fit(model, ckpt_path="last")  # Use ckpt_path="last" to resume if needed

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save the model
    model.save_to("base_multiblock_model.pt")
    print("Base model saved as 'base_multiblock_model.pt'.")

    # Print final losses if needed
    print(f"Final Training Losses: {loss_history.train_losses}")
    print(f"Final Validation Losses: {loss_history.val_losses}")

    # Plot final loss curves (log10 scale)
    plt.figure(figsize=(10, 6))
    print(loss_history.train_losses)
    print(loss_history.val_losses)
    if loss_history.train_losses:  # Check if list is not empty before plotting
        # Filter out potential NaNs or infinities if any resulted from log10(0)
        train_losses_log10 = np.log10(loss_history.train_losses)
        train_losses_log10 = train_losses_log10[np.isfinite(train_losses_log10)]
        plt.plot(train_losses_log10, label="Train Loss")

    if loss_history.val_losses:  # Check if list is not empty before plotting
        # Filter out potential NaNs or infinities
        val_losses_log10 = np.log10(loss_history.val_losses)
        val_losses_log10 = val_losses_log10[np.isfinite(val_losses_log10)]
        plt.plot(val_losses_log10, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("log10(Loss)")  # Updated label
    plt.grid(True)
    plt.legend()
    plt.title("log10(Loss) Curves over Epochs")  # Updated title
    plt.savefig("multiblock_loss.png")
    plt.close()
    print("Loss curves plot saved as 'multiblock_loss.png'.")

    # Perform final test evaluation run if test data is available
    if model.cfg.test_ds is not None:
        print("\nPerforming final evaluation on test data...")
        model.setup_test_data(model.cfg.test_ds)
        trainer.test(model)
    else:
        print("\nSkipping final test evaluation: Test data config not available.")
