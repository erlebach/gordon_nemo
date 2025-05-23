"""Multilayer Regression

Original version used the pytorch_lightning from Nemo.
This version uses the lightning module.
"""

import logging
import os
import time
from typing import Any, Dict, List, Union, cast

import matplotlib.pyplot as plt
import numpy as np

# old
import torch
import torch.nn as nn
from exp_lightning.multiblock_regressor_impl import (
    LossHistory,
    PredictionPlotter,
    # plot_loss_curves,
)
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# from torch.compile import
# disallow_in_graph = torch._dynamo.disallow_in_graph


# Add LoraLinear class before MLP
class LoraLinear(nn.Module):
    """A Linear layer with LoRA adaptation.

    Args:
        in_features: Input features.
        out_features: Output features.
        lora_rank: The rank of the low-rank adaptation matrices.
        lora_alpha: Scaling factor for the adapter output.
        lora_dropout: Dropout probability for the LoRA path.
        bias: Whether to use bias in the base linear layer (bias is not used in LoRA matrices).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        bias: bool = True,  # Keep bias for the base linear layer
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.lora_dropout = lora_dropout

        # Base linear layer (can be pretrained and potentially frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if lora_rank > 0:
            # LoRA A matrix (in_features -> lora_rank)
            self.lora_a = nn.Linear(in_features, lora_rank, bias=False)
            # LoRA B matrix (lora_rank -> out_features)
            self.lora_b = nn.Linear(
                lora_rank, out_features, bias=False
            )  # LoRA does not use bias

            # Optional dropout for the LoRA path
            self.dropout = nn.Dropout(p=lora_dropout)

            # Initialize LoRA weights
            # Initialize A with Kaiming uniform and B with zeros as per common practice
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            nn.init.zeros_(self.lora_b.weight)

            # Set LoRA parameters as trainable initially
            self.lora_a.weight.requires_grad = True
            self.lora_b.weight.requires_grad = True
            if bias and self.linear.bias is not None:
                self.linear.bias.requires_grad = (
                    True  # Keep bias trainable for base linear
                )
        else:
            # If lora_rank is 0, this is just a standard linear layer
            self.lora_a = None
            self.lora_b = None
            self.dropout = None
            # Ensure base linear weights are trainable if LoRA is off (default behavior)
            self.linear.weight.requires_grad = True

    def forward(self, x):
        # Calculate base linear output
        base_output = self.linear(x)

        if self.lora_rank > 0 and self.lora_a is not None and self.lora_b is not None:
            # Calculate LoRA output
            lora_output = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
            # Add LoRA output to base output
            return base_output + lora_output
        else:
            # If LoRA is disabled (rank 0), just return the base output
            return base_output


# Modify the existing MLP class to use LoraLinear and accept LoRA params
class MLP(torch.nn.Module):
    """MLP layer modified to use LoraLinear for LoRA adaptation.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        activation: Activation function to use.
        dropout: Dropout probability for the base MLP path.
        lora_rank: The rank of the low-rank adaptation matrices for LoRA layers.
        lora_alpha: Scaling factor for the LoRA adapter output.
        lora_dropout: Dropout probability for the LoRA paths.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "tanh",
        dropout: float = 0.0,  # Base MLP dropout
        lora_rank: int = 0,  # Default to no LoRA
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,  # LoRA specific dropout
    ):
        super().__init__()

        # Use LoraLinear for the linear layers, passing LoRA parameters
        self.fc1 = LoraLinear(
            input_dim,
            hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=True,  # Use bias for fc1
        )

        self.dropout1 = torch.nn.Dropout(dropout)  # Base MLP dropout

        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        # Use LoraLinear for the second linear layer, passing LoRA parameters
        self.fc2 = LoraLinear(
            hidden_dim,
            output_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=True,  # Use bias for fc2
        )

    def forward(self, x):
        # The forward pass remains simple because LoraLinear handles the addition
        x = self.fc1(x)
        x = self.dropout1(x)  # Apply base MLP dropout after fc1 + lora1
        x = self.activation(x)
        x = self.fc2(x)
        return x


# Modify ResidualMLP __init__ to accept and pass down LoRA params
class ResidualMLP(torch.nn.Module):
    """Residual MLP module using potentially LoRA-modified MLP layers.

    Args:
        dim: Feature dimension.
        num_layers: Number of MLP layers.
        hidden_dim: Hidden dimension for each MLP layer.
        activation: Activation function to use.
        dropout: Dropout probability for the base MLP path.
        lora_rank: LoRA rank passed to MLP layers.
        lora_alpha: LoRA alpha passed to MLP layers.
        lora_dropout: LoRA dropout passed to MLP layers.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: int,
        activation: str = "tanh",
        dropout: float = 0.1,
        lora_rank: int = 0,  # Add LoRA parameters
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Create layers using the modified MLP, passing LoRA parameters
        self.layers = nn.ModuleList(
            [
                MLP(
                    dim,
                    hidden_dim,
                    dim,  # Output dim of ResidualMLP internal layer is dim
                    activation,
                    dropout,
                    lora_rank=lora_rank,  # Pass LoRA params
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        input_x = x
        for layer in self.layers:
            x = layer(x)
            x = x + input_x  # Residual connection
            input_x = x
        return x


# Modify MultiBlockRegressor __init__ to accept and pass down LoRA params
class MultiBlockRegressor(torch.nn.Module):
    """A multi-block regressor (core architecture) using potentially LoRA-modified ResidualMLP blocks.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension for each block.
        output_dim: Output dimension.
        num_blocks: Number of residual blocks.
        num_layers_per_block: Number of MLP layers per block.
        activation: Activation function to use.
        dropout: Dropout probability for the base MLP path.
        lora_rank: LoRA rank passed to ResidualMLP blocks.
        lora_alpha: LoRA alpha passed to ResidualMLP blocks.
        lora_dropout: LoRA dropout passed to ResidualMLP blocks.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 1,
        num_blocks: int = 2,
        num_layers_per_block: int = 2,
        activation: str = "tanh",
        dropout: float = 0.1,
        lora_rank: int = 0,  # Add LoRA parameters
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.dropout = dropout  # Base MLP dropout
        self.lora_rank = lora_rank  # Store LoRA rank
        # Create the blocks using the modified ResidualMLP, passing LoRA parameters
        self.blocks = nn.ModuleList()

        # First block takes the input dimension
        first_block = ResidualMLP(
            dim=input_dim,
            num_layers=num_layers_per_block,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
            lora_rank=lora_rank,  # Pass LoRA params
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
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
                dropout=dropout,
                lora_rank=lora_rank,  # Pass LoRA params
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.blocks.append(block)
            setattr(self, f"block_{i}", block)

        # Final output projection if needed
        # Using standard Linear layer for output projection for simplicity, not adding LoRA here
        if input_dim != output_dim:
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
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


# Update MultiBlockRegressorPT __init__ to read LoRA parameters from config
class MultiBlockRegressorPT(LightningModule):
    """A multi-block regressor wrapped as a Lightning Module.

    Uses composition to contain a MultiBlockRegressor instance.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # First call LightningModule's init
        super().__init__()

        # Store the config received. This config is expected to contain
        # 'arch', 'optim', and potentially 'adapter' and dataset keys
        self.cfg = cfg

        # Access architectural parameters from cfg.arch
        arch_cfg = cfg.arch

        # Access LoRA parameters from cfg.adapter (if they exist)
        # Provide default values if adapter config is missing
        adapter_cfg = cfg.get(
            "adapter",
            OmegaConf.create({"lora_rank": 0, "lora_alpha": 32, "lora_dropout": 0.0}),
        )

        # Now create the regressor as a member variable
        # Pass the architectural and LoRA parameters down to the regressor
        self.regressor = MultiBlockRegressor(
            input_dim=arch_cfg.input_dim,
            hidden_dim=arch_cfg.hidden_dim,
            output_dim=arch_cfg.output_dim,
            num_blocks=arch_cfg.num_blocks,
            num_layers_per_block=arch_cfg.num_layers_per_block,
            activation=arch_cfg.activation,
            dropout=arch_cfg.dropout,  # Base MLP dropout
            # Pass LoRA specific parameters
            lora_rank=adapter_cfg.lora_rank,
            lora_alpha=adapter_cfg.lora_alpha,
            lora_dropout=adapter_cfg.lora_dropout,
        )

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Important: Save hyperparameters! This makes them appear in TensorBoard HPARAMS tab
        # Include adapter config in hyperparameters by saving the entire config
        self.save_hyperparameters(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate to the regressor's forward method
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Keep self.log calls - Lightning will send them to the logger
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Keep self.log calls - Lightning will send them to the logger
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Keep self.log calls - Lightning will send them to the logger
        self.log("test_loss", loss, prog_bar=True)
        return loss

    # Modify configure_optimizers to train only LoRA parameters and biases when LoRA is enabled
    def configure_optimizers(self):
        optim_cfg = self.cfg.optim

        # Determine if LoRA is enabled based on the lora_rank parameter in the config
        lora_enabled = (
            self.cfg.get("adapter", OmegaConf.create({"lora_rank": 0})).lora_rank > 0
        )

        if lora_enabled:
            print(
                "LoRA is enabled. Configuring optimizer to train only LoRA parameters and biases."
            )
            # Freeze all parameters initially
            for param in self.parameters():
                param.requires_grad = False

            # Unfreeze parameters that are part of the LoraLinear layers (lora_a, lora_b)
            # and the biases of the base linear layers within LoraLinear if bias is used
            trainable_params = []
            for name, param in self.named_parameters():
                # Check if the parameter belongs to a LoraLinear layer's lora_a or lora_b weights
                # or if it's the bias of the base linear layer within LoraLinear and requires grad
                if (
                    "lora_a" in name
                    or "lora_b" in name
                    or ("linear.bias" in name and param.requires_grad)
                ):
                    param.requires_grad = True  # Ensure they are trainable
                    trainable_params.append(param)
                    # print(f"Parameter '{name}' is trainable (LoRA or bias)")
                # else:
                # print(f"Parameter '{name}' is frozen")
            print(f"Number of trainable parameters: {len(trainable_params)}")
            if not trainable_params:
                logging.warning(
                    "No trainable parameters found when LoRA is enabled. Check model structure and parameter naming."
                )
        else:
            print(
                "LoRA is disabled. Configuring optimizer to train all model parameters."
            )
            # If LoRA is disabled, train all parameters of the model
            trainable_params = self.parameters()

        optimizer = torch.optim.Adam(
            trainable_params,  # Pass the determined set of trainable parameters
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )

        # Configure learning rate scheduler if specified
        if (
            hasattr(optim_cfg, "sched")
            and OmegaConf.select(optim_cfg, "sched.name") is not None
        ):
            scheduler_cfg = optim_cfg.sched
            if scheduler_cfg.name == "CosineAnnealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_cfg.T_max,
                    eta_min=scheduler_cfg.min_lr,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",  # Monitor validation loss for scheduling
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                raise NotImplementedError(
                    f"Scheduler {scheduler_cfg.name} not implemented"
                )
        else:
            return optimizer

    # Simple save and restore methods to mimic ModelPT
    # You can still keep this if you want to save the final model state manually
    def save_to(self, save_path):
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    seed_everything(42, workers=True)

    # Load the full configuration from the YAML file
    # config = OmegaConf.load("config/multiblock_config.yaml")
    config = OmegaConf.load("config/multiblock_config.yaml")

    # Get specific config sections
    # config_model = config.model # Not strictly needed as we pass the full config structure
    config_data = config.data
    config_trainer = config.trainer
    config_adapter = config.get("adapter", None)  # Get adapter config if it exists

    # Get the resume_from_checkpoint path from the TOP LEVEL of the config
    resume_from_checkpoint_path = config.get("resume_from_checkpoint")

    if resume_from_checkpoint_path is not None:
        print(
            f"Attempting to resume training from checkpoint: {resume_from_checkpoint_path}"
        )
        if not os.path.exists(resume_from_checkpoint_path):
            logging.warning(
                f"Checkpoint file not found at {resume_from_checkpoint_path}. Starting training from scratch."
            )
            resume_from_checkpoint_path = None  # Reset to None if file doesn't exist

    print(f"{resume_from_checkpoint_path=}")

    # Instantiate the DataModule - The trainer will call setup later
    data_module = MultiBlockRegressorDataModule(cfg=config_data)

    # Instantiate the Model - Needs arch, optim, and potentially adapter configs
    # Create a merged config structure for the model constructor.
    # The MultiBlockRegressorPT constructor expects 'arch', 'optim', and optionally 'adapter'
    # at the root level of the cfg passed to it.
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

    # Assign adapter config if it exists
    if config_adapter is not None:
        model_init_cfg.adapter = config_adapter
    else:
        # If no adapter config, ensure default LoRA parameters are used (lora_rank=0)
        model_init_cfg.adapter = OmegaConf.create(
            {"lora_rank": 0, "lora_alpha": 32, "lora_dropout": 0.0}
        )
        print("No 'adapter' section found in config. LoRA is disabled by default.")

    # Note: Data config is handled by the DataModule, not the ModelPT init in this structure.
    # Remove the checks and assignments for train_ds, validation_ds, test_ds to model_init_cfg
    # from the previous version of the __main__ block, as they are handled by the DataModule now.
    # This simplifies the model_init_cfg passed to MultiBlockRegressorPT.

    model = MultiBlockRegressorPT(cfg=model_init_cfg)  # Pass the merged config

    # Setup callbacks
    # LossHistory and PredictionPlotter remain useful.
    loss_history = LossHistory()

    # Setup prediction plotter callback - Ensure it gets the test data path from the main config
    test_data_path_for_plotter = None
    if (
        "data" in config  # Check main data config
        and "test_ds" in config.data
        and "file_path" in config.data.test_ds
    ):
        test_data_path_for_plotter = config.data.test_ds.file_path
    else:
        logging.warning(
            "Test data file path not found in config. Prediction plotting will be skipped."
        )

    prediction_plotter = None
    if test_data_path_for_plotter is not None:
        plot_interval_cfg = config_trainer.get("plot_interval", 2)
        prediction_plotter = PredictionPlotter(
            test_data_path=test_data_path_for_plotter,
            loss_history_callback=loss_history,
            plot_interval=plot_interval_cfg,
        )

    # Setup ModelCheckpoint callback - using config_trainer
    checkpoint_callback = None
    if "checkpoint" in config_trainer:
        checkpoint_cfg = config_trainer.checkpoint
        checkpoint_dirpath = checkpoint_cfg.get("dirpath", "checkpoints/")
        checkpoint_every_n_epochs = checkpoint_cfg.get("every_n_epochs")
        checkpoint_save_last = checkpoint_cfg.get("save_last", False)

        os.makedirs(checkpoint_dirpath, exist_ok=True)

        if checkpoint_every_n_epochs is not None:
            print(
                f"Setting up checkpointing every {checkpoint_every_n_epochs} epochs to {checkpoint_dirpath}"
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="{epoch}",
                every_n_epochs=checkpoint_every_n_epochs,
                save_last=checkpoint_save_last,
                save_top_k=-1,
            )
        elif checkpoint_save_last:
            print(f"Setting up checkpointing to save last only to {checkpoint_dirpath}")
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="last",
                save_last=True,
                save_top_k=0,
            )
        else:
            logging.warning(
                "Checkpoint configuration found but neither 'every_n_epochs' nor 'save_last' is specified. Checkpointing is disabled."
            )

    # Create TensorBoard Logger instance
    tensorboard_logger = TensorBoardLogger(
        "logs", name="multiblock_regressor_experiment"
    )
    print(f"TensorBoard logs will be saved to: {tensorboard_logger.log_dir}")

    # Collect all active callbacks
    callbacks_list = [loss_history]
    callbacks_list = cast(list[Callback], callbacks_list)
    if prediction_plotter is not None:
        callbacks_list.append(prediction_plotter)
    if checkpoint_callback is not None:
        callbacks_list.append(checkpoint_callback)

    # Setup profiler
    profiler = PyTorchProfiler(filename="prof.out")

    # Instantiate Trainer using config_trainer and collected callbacks/logger
    trainer = Trainer(
        profiler=profiler,
        max_epochs=config_trainer.get("max_epochs", 10),
        logger=tensorboard_logger,
        enable_progress_bar=config_trainer.get("enable_progress_bar", True),
        accelerator=config_trainer.get("accelerator", "auto"),
        devices=config_trainer.get("devices", "auto"),
        strategy=config_trainer.get("strategy", "auto"),
        precision=config_trainer.get("precision", 32),
        num_nodes=config_trainer.get("num_nodes", 1),
        log_every_n_steps=config_trainer.get("log_every_n_steps", 5),
        check_val_every_n_epoch=config_trainer.get("check_val_every_n_epoch", 1),
        benchmark=config_trainer.get("benchmark", False),
        callbacks=callbacks_list,
    )

    # Train the model
    start = time.time()
    # Use trainer.fit with the model, datamodule, and optional checkpoint path
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint_path,
    )

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Print final losses
    print(
        f"Final Training Losses: {loss_history.train_losses[-1] if loss_history.train_losses else 'N/A'}"
    )
    print(
        f"Final Validation Losses: {loss_history.val_losses[-1] if loss_history.val_losses else 'N/A'}"
    )

    # Perform final test evaluation run if test data is available
    # trainer.test will automatically use the test_dataloader from the datamodule
    trainer.test(model, datamodule=data_module)

    # Save the final model if a path is specified in the config
    if hasattr(config, "nemo_path") and config.nemo_path is not None:
        # Note: This saves the entire model state dictionary, including LoRA weights if present.
        # If you only want to save LoRA weights, you'd need custom logic here.
        model.save_to(config.nemo_path)
        logging.info(f"Model saved to {config.nemo_path}")
