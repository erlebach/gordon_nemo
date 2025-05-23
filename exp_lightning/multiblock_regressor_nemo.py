"""NeMo-compatible MultiBlock Regressor with LoRA support.

This module provides a NeMo-compatible implementation of the MultiBlock Regressor
that inherits from NeMo's ModelPT and supports LoRA adaptation. It follows NeMo
conventions and integrates with Hydra for configuration management.
"""

import os
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from exp_lightning.multiblock_regressor_nemo_impl import (
    create_nemo_callbacks,
    setup_nemo_logging,
)
from nemo.core import ModelPT, adapter_mixins
from nemo.core.classes.common import typecheck
from nemo.core.config import hydra_runner
from nemo.core.neural_types import NeuralType, RegressionValuesType
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class LoraLinear(nn.Module):
    """A Linear layer with LoRA adaptation and optional gating.

    Args:
        in_features: Input features.
        out_features: Output features.
        lora_rank: The rank of the low-rank adaptation matrices.
        lora_alpha: Scaling factor for the adapter output.
        lora_dropout: Dropout probability for the LoRA path.
        bias: Whether to use bias in the base linear layer.
        gating_function: Type of gating function ('identity' or 'sigmoid').

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        bias: bool = True,
        gating_function: str = "identity",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gating_function = gating_function.lower()

        # Base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.is_lora = lora_rank > 0

        if self.is_lora:
            self.scaling = lora_alpha / lora_rank

            # LoRA matrices
            self.lora_a = nn.Linear(in_features, lora_rank, bias=False)
            self.lora_b = nn.Linear(lora_rank, out_features, bias=False)

            # Dropout for LoRA path
            self.dropout = nn.Dropout(p=lora_dropout)

            # Gating mechanism
            if self.gating_function != "identity":
                # Gating layer: maps input features to output features dimension
                # Typically a linear layer followed by an activation
                self.gating_layer = nn.Linear(in_features, out_features)
                if self.gating_function == "sigmoid":
                    self.gating_activation = nn.Sigmoid()
                    print("==> Gating function: Sigmoid")
                # Add other gating functions here if needed
                else:
                    raise ValueError(
                        f"Unsupported gating_function: {gating_function}. "
                        "Supported: 'identity', 'sigmoid'."
                    )

            self.is_gating = (
                self.gating_function != "identity" and self.gating_layer is not None
            )

            # Initialize LoRA weights
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            nn.init.zeros_(self.lora_b.weight)

            # Initialize gating layer weights (optional, default init often fine)
            if self.gating_layer:
                nn.init.kaiming_uniform_(self.gating_layer.weight, a=5**0.5)
                if self.gating_layer.bias is not None:
                    nn.init.zeros_(self.gating_layer.bias)

            # Set trainability (LoRA weights and bias of base linear layer by default)
            # Gating layer weights are also trainable if gating is enabled
            self.lora_a.weight.requires_grad = True
            self.lora_b.weight.requires_grad = True
            if bias and self.linear.bias is not None:
                self.linear.bias.requires_grad = True
            if self.gating_layer:
                self.gating_layer.weight.requires_grad = True
                if self.gating_layer.bias is not None:
                    self.gating_layer.bias.requires_grad = True
        else:
            self.scaling = 1.0
            self.linear.weight.requires_grad = True
            self.is_lora = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the LoRA linear layer with optional gating.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """
        base_output = self.linear(x)

        if self.is_lora:
            lora_output = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

            # Apply gating
            if self.is_gating:
                # Compute gating values from the input x
                gating_values = self.gating_activation(self.gating_layer(x))
                # Apply Hadamard product
                lora_output = lora_output * gating_values

            return base_output + lora_output
        else:
            return base_output


class MLP(nn.Module):
    """MLP layer with optional LoRA adaptation and gating.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        activation: Activation function to use.
        dropout: Dropout probability for the base MLP path.
        lora_rank: The rank of the low-rank adaptation matrices.
        lora_alpha: Scaling factor for the LoRA adapter output.
        lora_dropout: Dropout probability for the LoRA paths.
        gating_function: Type of gating function for LoRA layers ('identity' or 'sigmoid').

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "tanh",
        dropout: float = 0.0,
        lora_rank: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        gating_function: str = "identity",
    ):
        super().__init__()

        self.fc1 = LoraLinear(
            input_dim,
            hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=True,
            gating_function=gating_function,
        )

        self.dropout1 = nn.Dropout(dropout)

        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        self.fc2 = LoraLinear(
            hidden_dim,
            output_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=True,
            gating_function=gating_function,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ResidualMLP(nn.Module):
    """Residual MLP module using potentially LoRA-modified MLP layers with gating.

    Args:
        dim: Feature dimension.
        num_layers: Number of MLP layers.
        hidden_dim: Hidden dimension for each MLP layer.
        activation: Activation function to use.
        dropout: Dropout probability for the base MLP path.
        lora_rank: LoRA rank passed to MLP layers.
        lora_alpha: LoRA alpha passed to MLP layers.
        lora_dropout: LoRA dropout passed to MLP layers.
        gating_function: Type of gating function for MLP layers.

    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: int,
        activation: str = "tanh",
        dropout: float = 0.1,
        lora_rank: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        gating_function: str = "identity",
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList(
            [
                MLP(
                    dim,
                    hidden_dim,
                    dim,
                    activation,
                    dropout,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    gating_function=gating_function,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the residual MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        input_x = x
        for layer in self.layers:
            x = layer(x)
            x = x + input_x  # Residual connection
            input_x = x
        return x


class MultiBlockRegressor(nn.Module):
    """A multi-block regressor core architecture with optional LoRA support and gating.

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
        gating_function: Type of gating function for ResidualMLP blocks.
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
        lora_rank: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        gating_function: str = "identity",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lora_rank = lora_rank
        self.gating_function = gating_function

        self.blocks = nn.ModuleList()

        # Create blocks
        for i in range(num_blocks):
            block = ResidualMLP(
                dim=input_dim,
                num_layers=num_layers_per_block,
                hidden_dim=hidden_dim,
                activation=activation,
                dropout=dropout,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                gating_function=gating_function,
            )
            self.blocks.append(block)
            setattr(self, f"block_{i}", block)

        # Final output projection
        if input_dim != output_dim:
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


class MultiBlockRegressorNeMo(ModelPT, adapter_mixins.AdapterModelPTMixin):
    """NeMo-compatible MultiBlock Regressor with LoRA support and gating.

    This class wraps the MultiBlockRegressor into NeMo's ModelPT framework,
    enabling it to work with Hydra configuration and NeMo's experiment management.

    Args:
        cfg: Configuration object.
        trainer: PyTorch Lightning trainer instance or None.

    """

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer | None = None):
        # Pass trainer to superclass only if it's not None
        super().__init__(cfg=cfg, trainer=trainer)

        # Extract configuration sections
        arch_cfg = cfg.arch
        adapter_cfg = cfg.get(
            "adapter",
            OmegaConf.create(
                {
                    "lora_rank": 0,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "gating_function": "identity",
                }
            ),
        )

        # Create the core regressor model
        self.regressor = MultiBlockRegressor(
            input_dim=arch_cfg.input_dim,
            hidden_dim=arch_cfg.hidden_dim,
            output_dim=arch_cfg.output_dim,
            num_blocks=arch_cfg.num_blocks,
            num_layers_per_block=arch_cfg.num_layers_per_block,
            activation=arch_cfg.activation,
            dropout=arch_cfg.dropout,
            lora_rank=adapter_cfg.lora_rank,
            lora_alpha=adapter_cfg.lora_alpha,
            lora_dropout=adapter_cfg.lora_dropout,
            gating_function=adapter_cfg.gating_function,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Setup adapters if needed
        self.setup_adapters()

        # Setup data loaders
        self._setup_dataloader_from_config(cfg)

    @property
    def input_types(self) -> dict[str, NeuralType]:
        """Define input neural types for NeMo.

        Returns:
            Dictionary mapping input names to neural types.
        """
        return {"x": NeuralType(("B", "T"), RegressionValuesType())}

    @property
    def output_types(self) -> dict[str, NeuralType]:
        """Define output neural types for NeMo.

        Returns:
            Dictionary mapping output names to neural types.
        """
        return {"y": NeuralType(("B", "T"), RegressionValuesType())}

    @typecheck()
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using the regressor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        x, y = batch
        y_hat = self(x=x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute a single validation step.

        Args:
            batch: Validation batch.
            batch_idx: Batch index.

        Returns:
            Validation loss.

        """
        x, y = batch
        y_hat = self(x=x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Execute a single test step.

        Args:
            batch: Test batch.
            batch_idx: Batch index.

        Returns:
            Test loss.
        """
        x, y = batch
        y_hat = self(x=x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """Configure the optimizer and optionally the learning rate scheduler.

        Returns:
            Optimizer or dictionary with optimizer and scheduler.

        """
        optim_cfg = self.cfg.optim

        # Determine if LoRA is enabled
        lora_enabled = (
            self.cfg.get("adapter", OmegaConf.create({"lora_rank": 0})).lora_rank > 0
        )

        if lora_enabled:
            logging.info("LoRA is enabled. Training only LoRA parameters and biases.")
            # Freeze all parameters initially
            for param in self.parameters():
                param.requires_grad = False

            # Unfreeze LoRA parameters and biases
            trainable_params = []
            for name, param in self.named_parameters():
                if (
                    "lora_a" in name
                    or "lora_b" in name
                    or ("linear.bias" in name and param.requires_grad)
                ):
                    param.requires_grad = True
                    trainable_params.append(param)

            logging.info(f"Number of trainable parameters: {len(trainable_params)}")
        else:
            logging.info("LoRA is disabled. Training all model parameters.")
            trainable_params = self.parameters()

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )

        # Configure scheduler if specified
        if hasattr(optim_cfg, "sched") and optim_cfg.sched.name is not None:
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
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }

        return optimizer

    def _setup_dataloader_from_config(self, cfg: DictConfig) -> None:
        """Set up data loaders from configuration.

        Args:
            cfg: Configuration object.

        """
        if hasattr(cfg, "train_ds"):
            self.setup_training_data(cfg.train_ds)
        if hasattr(cfg, "validation_ds"):
            self.setup_validation_data(cfg.validation_ds)
        if hasattr(cfg, "test_ds"):
            self.setup_test_data(cfg.test_ds)

    def setup_training_data(self, train_data_config: DictConfig | dict):
        """Set up training data loader.

        Args:
            train_data_config: Configuration for training data.

        """
        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )

    def setup_validation_data(self, val_data_config: DictConfig | dict):
        """Setup validation data loader.

        Args:
            val_data_config: Configuration for validation data.

        """
        self._validation_dl: DataLoader | None = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )

    def setup_test_data(self, test_data_config: DictConfig | dict):
        """Setup test data loader.

        Args:
            test_data_config: Configuration for test data.

        """
        self._test_dl: DataLoader | None = self._get_dataloader_from_config(
            test_data_config, shuffle=False
        )

    def _get_dataloader_from_config(
        self, config: DictConfig, shuffle: bool = False
    ) -> DataLoader | None:
        """Create a dataloader from configuration.

        Args:
            config: Configuration for the dataloader.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader instance or None if creation fails.
        """
        try:
            file_path = config.file_path
            batch_size = config.batch_size
            num_workers = config.get("num_workers", 0)
            pin_memory = config.get("pin_memory", False)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found at: {file_path}")

            data = np.load(file_path)
            x = data["x"]
            y = data["y"]

            dataset = TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            logging.info(f"DataLoader created successfully for {file_path}.")
            return dataloader

        except Exception as e:
            logging.error(f"Error creating dataloader from config: {e}")
            return None

    def train_dataloader(self) -> DataLoader | None:
        """Return the training dataloader."""
        return getattr(self, "_train_dl", None)

    def val_dataloader(self) -> DataLoader | None:
        """Return the validation dataloader."""
        return getattr(self, "_validation_dl", None)

    def test_dataloader(self) -> DataLoader | None:
        """Return the test dataloader."""
        return getattr(self, "_test_dl", None)

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List available pretrained models.

        Returns:
            List of available model names.
        """
        return []

    # Adapter mixin methods
    def add_adapter(self, name: str, cfg: DictConfig):
        """Add an adapter to the model.

        Args:
            name: Name of the adapter.
            cfg: Configuration for the adapter.
        """
        super().add_adapter(name, cfg)
        logging.info(f"Added adapter: {name}")

    def get_enabled_adapters(self) -> list[str]:
        """Get the list of enabled adapters.

        Returns:
            List of enabled adapter names.
        """
        return super().get_enabled_adapters()

    def is_adapter_available(self) -> bool:
        """Check if any adapter is available.

        Returns:
            True if at least one adapter is available.
        """
        return super().is_adapter_available()

    def set_enabled_adapters(self, name: str | None = None, enabled: bool = True):
        """Enable or disable adapters.

        Args:
            name: Name of the adapter to enable/disable, or None for all adapters.
            enabled: Whether to enable or disable the adapter(s).
        """
        super().set_enabled_adapters(name, enabled)

    @property
    def adapter_module_names(self) -> list[str]:
        """Get the list of adapter module names.

        Returns:
            List of module names that support adapters.
        """
        module_names = [""]
        for i in range(self.regressor.num_blocks):
            module_names.append(f"block_{i}")
        return module_names

    @property
    def default_adapter_module_name(self) -> str:
        """Get the default adapter module name.

        Returns:
            Name of the default adapter module.
        """
        return "block_0"

    def check_valid_model_with_adapter_support_(self):
        """Check if the model supports adapters."""
        pass


@hydra_runner(config_path="config", config_name="multiblock_nemo_config")
def main(cfg: DictConfig) -> None:
    """Execute training function using Hydra configuration.

    Args:
        cfg: Hydra configuration object.

    """
    # Setup logging
    setup_nemo_logging()
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if data files exist
    train_data_path = cfg.model.train_ds.file_path
    if not Path(train_data_path).exists():
        logging.error(f"Training data not found at {train_data_path}")
        logging.info("Please ensure data files are generated before training.")
        return

    # Create trainer
    # Trainer configuration comes from cfg.trainer
    trainer = pl.Trainer(**cfg.trainer)

    # Setup experiment manager
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Manually add ModelCheckpoint if needed (since we disabled automatic creation in config)
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="{epoch}-{val_loss:.3f}",
        save_last=True,
    )
    # Access callbacks attribute directly - seems the linter might be overly cautious here or expects a specific Trainer subclass
    trainer.callbacks.append(checkpoint_callback)

    # Create model
    model = MultiBlockRegressorNeMo(cfg=cfg.model, trainer=trainer)

    # Create custom callbacks
    test_data_path = cfg.model.get("test_ds", {}).get("file_path")
    # Pass the full DictConfig to create_nemo_callbacks
    custom_callbacks = create_nemo_callbacks(cfg, test_data_path)

    # Add custom callbacks to trainer
    for callback in custom_callbacks:
        # Access callbacks attribute directly
        trainer.callbacks.append(callback)

    # Training
    logging.info("Starting training...")
    start_time = time.time()
    trainer.fit(model)
    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

    # Testing
    if test_data_path and os.path.exists(test_data_path):
        logging.info("Running testing...")
        trainer.test(model)

    # Save the model if specified
    if hasattr(cfg, "nemo_path") and cfg.nemo_path is not None:
        model.save_to(cfg.nemo_path)
        logging.info(f"Model saved to {cfg.nemo_path}")


if __name__ == "__main__":
    main()
