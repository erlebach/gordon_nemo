"""Multilayer Regression

Original version used the pytorch_lightning from Nemo.
This version uses the lightning module.
"""

import logging
import os
import time
from typing import Any, Dict, List, Union

import numpy as np

# old
import torch
import torch.nn as nn
from exp_lightning.multiblock_regressor_impl import (
    LossHistory,
    PredictionPlotter,
    plot_loss_curves,
)
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


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
    """A LightningDataModule for the MultiBlockRegressor."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        """Set up data loaders for different stages (fit, train, val, test).

        This method is called by the Lightning Trainer.

        Args:
            stage: The stage of the training process (e.g., 'fit', 'train',
                'validate', 'test', 'predict'). Defaults to None.

        """
        # The logic inside setup remains the same, it responds to the stage
        if stage in ("fit", "train", None):
            # Check if config for train_ds and validation_ds exists before setting up
            if self.cfg and hasattr(self.cfg, "train_ds"):
                self._setup_training_data(self.cfg.train_ds)
            else:
                logging.warning(
                    "Train dataset config not found or incomplete. Cannot setup training dataloader."
                )
                self._train_dl = None  # Ensure dataloader is None if config is missing

            if self.cfg and hasattr(self.cfg, "validation_ds"):
                self._setup_validation_data(self.cfg.validation_ds)
            else:
                logging.warning(
                    "Validation dataset config not found or incomplete. Cannot setup validation dataloader."
                )
                self._validation_dl = (
                    None  # Ensure dataloader is None if config is missing
                )

        if stage == "test":
            # Check if config for test_ds exists before setting up
            if self.cfg and hasattr(self.cfg, "test_ds"):
                self._setup_test_data(self.cfg.test_ds)
            else:
                logging.warning(
                    "Test dataset config not found or incomplete. Cannot setup test dataloader."
                )
                self._test_dl = None  # Ensure dataloader is None if config is missing

    def _setup_training_data(self, train_data_config: DictConfig | dict) -> None:
        """Set up the training dataloader. Internal method."""
        # Added explicit type check/assertion as discussed
        assert (
            train_data_config is not None
        ), "Training data configuration cannot be None."
        assert isinstance(
            train_data_config, DictConfig | dict
        ), f"Expected DictConfig or dict for training data config, but got {type(train_data_config)}"

        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )
        if self._train_dl:
            logging.info("Train DataLoader prepared")

    def _setup_validation_data(self, val_data_config: Union[DictConfig, Dict]) -> None:
        """Set up the validation dataloader. Internal method."""
        # Added explicit type check/assertion
        # Validation config can be None if not doing validation, so handle that first
        if val_data_config is None:
            self._validation_dl = None
            logging.info(
                "Validation data configuration is None. Skipping validation dataloader setup."
            )
            return

        assert isinstance(
            val_data_config, DictConfig | Dict
        ), f"Expected DictConfig or Dict for validation data config, but got {type(val_data_config)}"

        self._validation_dl = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )
        if self._validation_dl:
            logging.info("Validation DataLoader prepared")

    def _setup_test_data(self, test_data_config: DictConfig | dict) -> None:
        """Set up the test dataloader. Internal method."""
        # Added explicit type check/assertion
        # Test config can be None if not doing testing, so handle that first
        if test_data_config is None:
            self._test_dl = None
            logging.info(
                "Test data configuration is None. Skipping test dataloader setup."
            )
            return

        assert isinstance(
            test_data_config, DictConfig | dict
        ), f"Expected DictConfig or Dict for test data config, but got {type(test_data_config)}"

        self._test_dl = self._get_dataloader_from_config(
            test_data_config, shuffle=False
        )
        if self._test_dl:
            logging.info("Test DataLoader prepared")

    def _get_dataloader_from_config(
        self,
        config: DictConfig | dict,
        shuffle: bool = False,
    ) -> DataLoader | None:
        """Construct the dataloader from the config.

        The config should contain the following keys:
        - file_path: The path to the data file.
        - batch_size: The batch size.
        - num_workers: The number of workers to use.
        - pin_memory: Whether to pin memory.

        Args:
            config: The config to use to construct the dataloader.
            shuffle: Whether to shuffle the data.

        Returns:
            The dataloader, or None if config is None or loading fails.

        Raises:
            ValueError: If essential configuration keys ('file_path', 'batch_size') are missing or None.
            Exception: Propagates exceptions from data loading or dataloader creation.
        """
        # config is already checked for None by the calling _setup_ methods

        try:
            # Added checks for file_path and batch_size presence using .get() for safety
            file_path = config.get("file_path")
            batch_size = config.get("batch_size")

            # Use explicit check and raise ValueError if essential keys are missing or None
            if file_path is None:
                raise ValueError(
                    "Dataset config is missing or has None value for 'file_path'."
                )
            if batch_size is None:
                raise ValueError(
                    "Dataset config is missing or has None value for 'batch_size'."
                )

            # Add asserts after getting to help type checkers
            assert isinstance(
                file_path, str
            ), f"Expected 'file_path' to be a string, but got {type(file_path)}"
            assert isinstance(
                batch_size, int
            ), f"Expected 'batch_size' to be an integer, but got {type(batch_size)}"

            num_workers = config.get("num_workers", 0)
            pin_memory = config.get("pin_memory", False)

            # Add asserts for optional args if strict type checking is desired
            # assert isinstance(num_workers, int), f"Expected 'num_workers' to be an integer, but got {type(num_workers)}"
            # assert isinstance(pin_memory, bool), f"Expected 'pin_memory' to be a boolean, but got {type(pin_memory)}"

            print(f"Loading data from: {file_path}")
            # Use os.path.exists for robustness
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found at: {file_path}")

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
            # Log the error clearly
            logging.error(f"Error creating dataloader from config: {e}")
            # Re-raise the exception to signal failure
            raise e

    def train_dataloader(self) -> DataLoader | None:
        """Return training dataloader. Called by the trainer."""
        return self._train_dl

    def val_dataloader(self) -> DataLoader | None:
        """Return validation dataloader. Called by the trainer."""
        return self._validation_dl

    def test_dataloader(self) -> DataLoader | None:
        """Return test dataloader. Called by the trainer."""
        return self._test_dl


# Inherit directly from LightningModule and implement methods needed
class MultiBlockRegressorPT(LightningModule):
    """A multi-block regressor wrapped as a Lightning Module.

    Uses composition to contain a MultiBlockRegressor instance.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
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

    # Simple save and restore methods to mimic ModelPT
    def save_to(self, save_path):
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    seed_everything(42, workers=True)

    # Load the full configuration from the YAML file
    config = OmegaConf.load("config/multiblock_config.yaml")

    # Assuming config structure is trainer:{..}, data:{..}, model:{..}
    # Pass the relevant parts of the config to the DataModule and Model
    config_model = config.model
    config_data = config.data
    config_trainer = config.trainer

    # Instantiate the DataModule - The trainer will call setup later
    data_module = MultiBlockRegressorDataModule(cfg=config_data)
    # DO NOT CALL setup() MANUALLY HERE: data_module.setup("fit")

    # Instantiate the Model - Needs arch, optim, and potentially dataset configs
    # Create a merged config for the model constructor if its __init__ or methods
    # require arch, optim, and dataset configs at the top level of its cfg.
    # Based on the MultiBlockRegressorPT.__init__, it *does* expect arch at top level.
    # And setup_..._data methods expect dataset configs. So merge is appropriate.
    model_init_cfg = OmegaConf.create()

    # Check and assign arch config
    if "model" in config and "arch" in config.model:
        model_init_cfg.arch = config.model.arch
    else:
        raise ValueError("Config must contain 'model.arch' for model initialization.")

    # Check and assign optim config
    if "model" in config and "optim" in config.model:
        model_init_cfg.optim = config.model.optim
    else:
        logging.warning(
            "Warning: 'model.optim' not found in config. Optimizer configuration might be incomplete."
        )
        model_init_cfg.optim = OmegaConf.create(
            {}
        )  # Provide empty dict config if optional

    # Check and assign dataset configs to model_init_cfg if the model needs them internally
    # (The DataModule is the primary source for the trainer, but the model might use these too)
    if "data" in config:
        if "train_ds" in config.data:
            model_init_cfg.train_ds = config.data.train_ds
        else:
            logging.warning(
                "Warning: 'data.train_ds' not found in config. Model's internal training data setup might fail."
            )
            model_init_cfg.train_ds = None  # Set to None if missing

        if "validation_ds" in config.data:
            model_init_cfg.validation_ds = config.data.validation_ds
        else:
            logging.warning(
                "Warning: 'data.validation_ds' not found in config. Model's internal validation data setup might fail."
            )
            model_init_cfg.validation_ds = None  # Set to None if missing

        if "test_ds" in config.data:
            model_init_cfg.test_ds = config.data.test_ds
        else:
            logging.warning(
                "Warning: 'data.test_ds' not found in config. Model's internal test data setup might fail."
            )
            model_init_cfg.test_ds = None  # Set to None if missing
    else:
        logging.warning(
            "Warning: 'data' section not found in config. Model's internal data setup might fail."
        )
        # Provide empty dict configs for datasets if the section is missing entirely
        model_init_cfg.train_ds = None
        model_init_cfg.validation_ds = None
        model_init_cfg.test_ds = None

    model = MultiBlockRegressorPT(cfg=model_init_cfg)  # Pass the merged config

    # Setup loss history callback FIRST
    loss_history = LossHistory()

    # Setup prediction plotter callback
    # Pass the file path to the test data config AND the loss_history instance
    # Check if test_ds exists and has file_path before creating plotter
    test_data_path_for_plotter = None
    if (
        "data" in config
        and "test_ds" in config.data
        and "file_path" in config.data.test_ds
    ):
        test_data_path_for_plotter = config.data.test_ds.file_path
    else:
        logging.warning(
            "Test data file path not found in config. Prediction plotting will be skipped."
        )

    prediction_plotter = None  # Initialize to None
    if test_data_path_for_plotter is not None:
        prediction_plotter = PredictionPlotter(
            test_data_path=test_data_path_for_plotter,
            loss_history_callback=loss_history,  # Pass the LossHistory instance
            plot_interval=config_trainer.get(
                "plot_interval", 2
            ),  # Use .get() for plot_interval too
        )

    # Setup ModelCheckpoint callback
    checkpoint_callback = None  # Initialize to None
    if "checkpoint" in config_trainer:
        checkpoint_cfg = config_trainer.checkpoint
        checkpoint_dirpath = checkpoint_cfg.get("dirpath", "checkpoints/")
        checkpoint_every_n_epochs = checkpoint_cfg.get("every_n_epochs")
        checkpoint_save_last = checkpoint_cfg.get(
            "save_last", False
        )  # Default to False if not specified

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dirpath, exist_ok=True)

        # Create the ModelCheckpoint callback instance
        print(f"... {checkpoint_every_n_epochs=}")
        if checkpoint_every_n_epochs is not None:
            print(
                f"Setting up checkpointing every {checkpoint_every_n_epochs} epochs to {checkpoint_dirpath}"
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="{epoch}",  # Include epoch in filename
                every_n_epochs=checkpoint_every_n_epochs,
                save_last=checkpoint_save_last,
                save_top_k=-1,  # Save all checkpoints when using every_n_epochs
                # monitor='val_loss',  # Not needed for saving every N epochs
                # mode='min',
            )
        elif checkpoint_save_last:
            print(f"Setting up checkpointing to save last only to {checkpoint_dirpath}")
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="{epoch}-last",  # Filename for the last checkpoint
                save_last=True,
                save_top_k=0,  # Do not save top k based on a monitor
            )
        else:
            logging.warning(
                "Checkpoint configuration found but neither 'every_n_epochs' nor 'save_last' is specified. Checkpointing is disabled."
            )

    # Create trainer
    # Check if 'trainer' and 'max_epochs' exist
    max_epochs = config_trainer.get("max_epochs", 10)  # Use .get() for safety
    enable_checkpointing = config_trainer.get("enable_checkpointing", False)

    # Collect callbacks into a list, including the plotter only if it was created
    callbacks_list = [loss_history]
    if prediction_plotter is not None:
        callbacks_list.append(prediction_plotter)
    if checkpoint_callback is not None and enable_checkpointing is True:
        callbacks_list.append(checkpoint_callback)

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=enable_checkpointing,  # Consider enabling checkpointing
        enable_progress_bar=True,
        accelerator="cpu",  # Change to 'gpu' or 'auto' if GPU is available
        devices=1,  # Set to 'auto' or number of GPUs if using GPU
        callbacks=callbacks_list,  # Use the collected list of callbacks
    )

    # Train the model
    start = time.time()

    # Start training
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path="last",  # trainer gets data from datamodule
    )

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save the model
    model.save_to("base_multiblock_model.pt")
    print("Base model saved as 'base_multiblock_model.pt'.")

    # Print final losses if needed
    print(f"Final Training Losses: {loss_history.train_losses}")
    print(f"Final Validation Losses: {loss_history.val_losses}")

    plot_loss_curves(loss_history)

    print("\nPerforming final evaluation on test data...")
    trainer.test(model, datamodule=data_module)
