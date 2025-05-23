"""Multiblock Regressor: Adaptor Module

Implementation of LORA from scratch. The MLP network:
    self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

will be changed to:

    def forward(self, x):
        x = self.fc1(x) + self.adapt1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x) + self.adapt2(x)
        return x

where

    self.adapt1(x) = W1 * W2 * x

where W1 and W2 are two matrices or rank r. The adapter must be written in terms of the base class.

Base class: `MultiBlockRegressor` with constructor:

MultiBlockRegressor(self, input_dim: int = 1, hidden_dim: int = 16, output_dim: int = 1,
        num_blocks: int = 2, num_layers_per_block: int = 2, activation: str = "tanh",
        dropout: float = 0.1,)

Needed: redefine Linear modules fc1, fc2 with weight replaced by W1 * W2
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from jaxtyping import Float
from multiblock_regression.multiblock_regressor import (
    MLP,
    MultiBlockRegressor,
    ResidualMLP,
)
from nemo.core import ModelPT, adapter_mixins
from nemo.core.classes.common import typecheck
from nemo.core.config import hydra_runner
from nemo.core.neural_types import NeuralType, RegressionValuesType
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


# Simplified PEFT base class - implements locally since we can't rely on NeMo's implementation
class PEFT:
    def __init__(self):
        pass

    def transform(self, module, name=None, prefix=None):
        """Transform a module. To be implemented by subclasses."""
        return module

    def __call__(self, model_):
        """Apply transformations to model."""
        return model_


# Simplified AdapterWrapper - implements locally
class AdapterWrapper(nn.Module):
    def __init__(self, to_wrap, adapter):
        super().__init__()
        self.base_module = to_wrap
        self.adapter = adapter


class LinearAdapter(nn.Module):
    """Low-rank adapter for linear layers.

    Args:
        in_features: Input features.
        out_features: Output features.
        dim: Dimension of the low-rank projection.
        alpha: Scaling factor for the adapter output.
        dropout: Dropout probability.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.scale = alpha / dim
        self.in_features = in_features
        self.out_features = out_features

        # Low-rank projection: in_features -> dim -> out_features
        self.lora_a = nn.Linear(in_features, dim, bias=False)
        self.lora_b = nn.Linear(dim, out_features, bias=bias)

        # Initialize LoRA weights - A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Apply dropout
        x = self.dropout(x)

        # LoRA forward pass
        x = self.lora_a(x)  # Project to low-rank space
        x = self.lora_b(x)  # Project back to output space

        # Scale output
        return x * self.scale


class MLPAdapter(nn.Module):
    """Adapter for MLP modules.

    Args:
        in_features: Input features.
        hidden_features: Hidden features.
        out_features: Output features.
        dim: Adapter dimension.
        alpha: Adapter scaling factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1_adapter = LinearAdapter(
            in_features=in_features,
            out_features=hidden_features,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
        )

        self.fc2_adapter = LinearAdapter(
            in_features=hidden_features,
            out_features=out_features,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
        )

    def forward_fc1(self, x):
        return self.fc1_adapter(x)

    def forward_fc2(self, x):
        return self.fc2_adapter(x)

    # ... rest of the code remains unchanged ...

    def transform(self, m: nn.Module, name=None, prefix=None):
        """Apply LoRA transformation to modules.

        Args:
            m: The module to transform.
            name: The name of the module.
            prefix: The prefix path of the module.

        Returns:
            The transformed module if applicable, otherwise the original module.
        """
        if not isinstance(m, ResidualMLP):
            return m

        # Create a base adapter for this block
        adapter = ResidualMLPAdapter(
            dim=m.dim,
            hidden_dim=m.hidden_dim,
            num_layers=m.num_layers,
            adapter_dim=self.config.dim,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
        )

        return ResidualMLPLoraWrapper(to_wrap=m, adapter=adapter)

    def __call__(self, model: Union[MultiBlockRegressor, ModelPT]):
        """Apply LoRA adapters to the model.

        Args:
            model: MultiBlockRegressor or ModelPT model.

        Returns:
            The model with LoRA adapters applied.
        """
        # If ModelPT is provided, extract the actual MultiBlockRegressor
        if isinstance(model, ModelPT):
            if hasattr(model, "model"):
                actual_model = model.model
            else:
                actual_model = model
        else:
            actual_model = model

        if not isinstance(actual_model, MultiBlockRegressor):
            raise ValueError(f"Expected MultiBlockRegressor, got {type(actual_model)}")

        # Determine which blocks to adapt
        target_blocks = self.config.target_blocks
        if target_blocks is None:
            target_blocks = list(range(actual_model.num_blocks))

        # Apply adapters to selected blocks
        for i in target_blocks:
            if i < 0 or i >= actual_model.num_blocks:
                continue

            block_name = f"block_{i}"
            if not hasattr(actual_model, block_name):
                continue

            block = getattr(actual_model, block_name)
            wrapped_block = self.transform(block)

            # Replace the original block with the wrapped one
            setattr(actual_model, block_name, wrapped_block)
            # Also update in the ModuleList
            actual_model.blocks[i] = wrapped_block

        # Return the modified model
        return model


class MultiBlockRegressorModel(ModelPT, adapter_mixins.AdapterModelPTMixin):
    """MultiBlockRegressor model with NeMo ModelPT and adapter support.

    This class wraps the MultiBlockRegressor into NeMo's ModelPT framework,
    which allows it to be used with Hydra configuration and supports adapter-based
    fine-tuning.

    Args:
        cfg: Configuration object
        trainer: PyTorch Lightning trainer instance
    """

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # Initialize parent classes
        super().__init__(cfg=cfg, trainer=trainer)

        # Create the base regressor model using the arch namespace instead of model
        self.regressor = MultiBlockRegressor(
            input_dim=cfg.arch.input_dim,
            hidden_dim=cfg.arch.hidden_dim,
            output_dim=cfg.arch.output_dim,
            num_blocks=cfg.arch.num_blocks,
            num_layers_per_block=cfg.arch.num_layers_per_block,
            activation=cfg.arch.activation,
        )

        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Setup adapters if needed
        self.setup_adapters()

        # Setup dataloaders if configs are available
        if hasattr(cfg, "model"):
            if hasattr(cfg.model, "train_ds"):
                self.setup_training_data(cfg.model.train_ds)
            if hasattr(cfg.model, "validation_ds"):
                self.setup_validation_data(cfg.model.validation_ds)
            if hasattr(cfg.model, "test_ds"):
                self.setup_test_data(cfg.model.test_ds)

    @property
    def input_types(self):
        return {"x": NeuralType(("B", "T"), RegressionValuesType())}

    @property
    def output_types(self):
        return {"y": NeuralType(("B", "T"), RegressionValuesType())}

    def forward(self, x):
        """Forward pass using the regressor"""
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        y_hat = self(x=x)  # Use keyword argument
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        y_hat = self(x=x)  # Use keyword argument
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        y_hat = self(x=x)  # Use keyword argument
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer.

        Returns:
            The configured optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.get("weight_decay", 0.0),
        )

        # Configure learning rate scheduler if specified
        if hasattr(self.cfg.optim, "sched"):
            scheduler_config = self.cfg.optim.sched
            if scheduler_config.name == "CosineAnnealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.T_max,
                    eta_min=scheduler_config.min_lr,
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

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Setup training data loader.

        Args:
            train_data_config: Configuration for training data
        """
        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup validation data loader.

        Args:
            val_data_config: Configuration for validation data
        """
        self._validation_dl = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup test data loader.

        Args:
            test_data_config: Configuration for test data
        """
        self._test_dl = self._get_dataloader_from_config(
            test_data_config, shuffle=False
        )

    def _get_dataloader_from_config(
        self, config: Union[DictConfig, Dict], shuffle: bool = False
    ):
        """Helper method to create a dataloader from config.

        Args:
            config: Configuration for the dataloader
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader instance
        """
        import numpy as np

        # Load data from numpy file
        data = np.load(config.file_path)
        x = data["x"]
        y = data["y"]

        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )

        # Create dataloader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
        )

    def add_adapter(self, name: str, cfg: Union[DictConfig, Dict]):
        """Add an adapter to the model.

        Args:
            name: Name of the adapter
            cfg: Configuration for the adapter
        """
        super().add_adapter(name, cfg)

        if name.startswith("lora"):
            # Apply LoRA adapter
            lora_config = MultiBlockRegressorLoraConfig(
                dim=cfg.dim,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
                target_blocks=cfg.get("target_blocks", None),
            )

            # Create and apply the LoRA adapter
            lora = MultiBlockRegressorLora(lora_config)
            lora(self.regressor)

    def get_enabled_adapters(self):
        """Get the list of enabled adapters.

        Returns:
            List of enabled adapter names
        """
        return super().get_enabled_adapters()

    def is_adapter_available(self) -> bool:
        """Check if any adapter is available.

        Returns:
            True if at least one adapter is available
        """
        return super().is_adapter_available()

    def set_enabled_adapters(self, name=None, enabled=True):
        """Enable or disable adapters.

        Args:
            name: Name of the adapter to enable/disable, or None for all adapters
            enabled: Whether to enable or disable the adapter(s)
        """
        super().set_enabled_adapters(name, enabled)

    @property
    def adapter_module_names(self) -> List[str]:
        """Get the list of adapter module names.

        Returns:
            List of module names that support adapters
        """
        module_names = [""]
        for i in range(self.regressor.num_blocks):
            module_names.append(f"block_{i}")
        return module_names

    @property
    def default_adapter_module_name(self) -> str:
        """Get the default adapter module name.

        Returns:
            Name of the default adapter module
        """
        return "block_0"

    def check_valid_model_with_adapter_support_(self):
        """Check if the model supports adapters."""
        pass

    @classmethod
    def list_available_models(cls):
        """
        This method is required by ModelPT and should return a list of
        pretrained model weights available for this class.

        Returns:
            List of pretrained models available for this class
        """
        return []

    def train_dataloader(self):
        """Return the training dataloader with better error handling."""
        if not hasattr(self, "_train_dl") or self._train_dl is None:
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "train_ds"):
                print(
                    f"Setting up training dataloader from {self.cfg.model.train_ds.file_path}"
                )
                self.setup_training_data(self.cfg.model.train_ds)
            else:
                raise ValueError("No training data configuration found in model config")

        if self._train_dl is None:
            raise ValueError("Failed to create training dataloader")

        return self._train_dl

    def val_dataloader(self):
        """Return the validation dataloader with better error handling."""
        if not hasattr(self, "_validation_dl") or self._validation_dl is None:
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "validation_ds"):
                print(
                    f"Setting up validation dataloader from {self.cfg.model.validation_ds.file_path}"
                )
                self.setup_validation_data(self.cfg.model.validation_ds)
            else:
                raise ValueError(
                    "No validation data configuration found in model config"
                )

        if self._validation_dl is None:
            raise ValueError("Failed to create validation dataloader")

        return self._validation_dl

    def test_dataloader(self):
        """Return the test dataloader with better error handling."""
        if not hasattr(self, "_test_dl") or self._test_dl is None:
            if hasattr(self.cfg, "model") and hasattr(self.cfg, "test_ds"):
                print(
                    f"Setting up test dataloader from {self.cfg.model.test_ds.file_path}"
                )
                self.setup_test_data(self.cfg.model.test_ds)
            else:
                raise ValueError("No test data configuration found in model config")

        if self._test_dl is None:
            raise ValueError("Failed to create test dataloader")

        return self._test_dl

    @classmethod
    def restore_from(
        cls,
        restore_path,
        trainer=None,
        override_config_path=None,
        map_location=None,
        strict=True,
    ):
        """Override restore_from to properly handle config and dataloaders after restoration."""
        # Call the parent restore_from method
        model = super().restore_from(
            restore_path=restore_path,
            trainer=trainer,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=strict,
        )

        # Ensure the model knows it's been restored and needs to check dataloaders
        model._dataloaders_initialized = False

        return model


@hydra_runner(config_path=".", config_name="multiblock_fixed_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if data needs to be prepared
    import os

    if not os.path.exists("data/sine_train.npz"):
        logging.info("Data files not found. Creating datasets...")
        prepare_data()

    # Make sure Trainer settings don't conflict with exp_manager
    if "logger" in cfg.trainer and cfg.trainer.logger:
        cfg.trainer.logger = False

    if "enable_checkpointing" in cfg.trainer and cfg.trainer.enable_checkpointing:
        cfg.trainer.enable_checkpointing = False

    # Set up trainer with Lightning
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Create model from checkpoint
    model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.restore_path, trainer=trainer
    )

    # Debug print
    print(f"Model restored from: {cfg.model.restore_path}")

    # Set up dataloaders explicitly
    if hasattr(cfg.model, "train_ds"):
        print(f"Setting up training data from: {cfg.model.train_ds.file_path}")
        try:
            model.setup_training_data(cfg.model.train_ds)
            print("Training data setup successful")
        except Exception as e:
            print(f"Error setting up training data: {e}")

    if hasattr(cfg.model, "validation_ds"):
        print(f"Setting up validation data from: {cfg.model.validation_ds.file_path}")
        try:
            model.setup_validation_data(cfg.model.validation_ds)
            print("Validation data setup successful")
        except Exception as e:
            print(f"Error setting up validation data: {e}")

    if hasattr(cfg.model, "test_ds"):
        print(f"Setting up test data from: {cfg.model.test_ds.file_path}")
        try:
            model.setup_test_data(cfg.model.test_ds)
            print("Test data setup successful")
        except Exception as e:
            print(f"Error setting up test data: {e}")

    # Try to verify dataloaders are created
    print(f"Training dataloader: {model._train_dl is not None}")
    print(f"Validation dataloader: {model._validation_dl is not None}")
    print(f"Test dataloader: {model._test_dl is not None}")

    # Add and enable the adapter
    if hasattr(cfg, "adapter"):
        adapter_cfg = cfg.adapter
        model.add_adapter(adapter_cfg.name, adapter_cfg)
        model.set_enabled_adapters(adapter_cfg.name, enabled=True)

        # Freeze base model parameters
        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    # Training
    logging.info("Starting training...")
    trainer.fit(model)
    logging.info("Training completed.")

    # Testing
    if hasattr(cfg, "test_ds") and cfg.test_ds.file_path is not None:
        logging.info("Running testing...")
        trainer.test(model)

    # Save the model if a path is specified
    if hasattr(cfg, "nemo_path") and cfg.nemo_path is not None:
        model.save_to(cfg.nemo_path)
        logging.info(f"Model saved to {cfg.nemo_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    seed_everything(42, workers=True)

    # Load the full configuration from the YAML file
    config = OmegaConf.load("config/multiblock_config.yaml")

    # Get specific config sections
    config_model = config.model
    config_data = config.data
    config_trainer = config.trainer

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

    # Instantiate the Model - Needs arch, optim, and potentially dataset configs
    # Create a merged config for the model constructor if its __init__ or methods
    # require arch, optim, and dataset configs at the top level of its cfg.
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
    # Note: With DataModule, the model typically doesn't need data config internally.
    # Consider if this is necessary based on your ModelPT class design.
    # If ModelPT does not need data config, you can remove this section.
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
    # Keep LossHistory if you still want to access losses as a list after training,
    # but logging will now primarily be handled by TensorBoard.
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
        # Use .get() for plot_interval from config_trainer
        plot_interval_cfg = config_trainer.get("plot_interval", 2)
        prediction_plotter = PredictionPlotter(
            test_data_path=test_data_path_for_plotter,
            loss_history_callback=loss_history,  # Pass the LossHistory instance
            plot_interval=plot_interval_cfg,
        )

    # Setup ModelCheckpoint callback - Keep this for saving model checkpoints
    checkpoint_callback = None  # Initialize to None
    # Check if 'checkpoint' section exists under trainer
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
            )
        elif checkpoint_save_last:
            print(f"Setting up checkpointing to save last only to {checkpoint_dirpath}")
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename="last",  # Standard filename for the last checkpoint
                save_last=True,
                save_top_k=0,  # Do not save top k based on a monitor
            )
        else:
            logging.warning(
                "Checkpoint configuration found but neither 'every_n_epochs' nor 'save_last' is specified. Checkpointing is disabled."
            )

    # Create TensorBoard Logger instance
    # Logs will be saved to ./logs/tensorboard/multiblock_regressor_experiment/version_x
    tensorboard_logger = TensorBoardLogger(
        "logs", name="multiblock_regressor_experiment"
    )
    print(f"TensorBoard logs will be saved to: {tensorboard_logger.log_dir}")

    # Collect all active callbacks into a list
    callbacks_list = [
        loss_history
    ]  # Keep LossHistory if needed for PredictionPlotter or post-train analysis
    callbacks_list = cast(list[Callback], callbacks_list)
    if prediction_plotter is not None:
        callbacks_list.append(prediction_plotter)
    if checkpoint_callback is not None:
        callbacks_list.append(checkpoint_callback)

    # profiler = PyTorchProfiler()
    profiler = PyTorchProfiler(filename="prof.out")
    # Requires CUDA and nvcc
    # profiler = PyTorchProfiler(emit_nvtx=True)

    trainer = Trainer(
        profiler=profiler,
        max_epochs=config_trainer.get("max_epochs", 10),
        # logger=False, # REMOVE THIS! Let the logger handle logging
        logger=tensorboard_logger,  # Pass the TensorBoard logger
        enable_progress_bar=True,
        accelerator=config_trainer.get(
            "accelerator", "auto"
        ),  # Use .get() for accelerator and devices
        devices=config_trainer.get("devices", "auto"),
        strategy=config_trainer.get(
            "strategy", "auto"
        ),  # Use .get() for strategy and precision
        precision=config_trainer.get("precision", 32),
        num_nodes=config_trainer.get("num_nodes", 1),
        log_every_n_steps=config_trainer.get("log_every_n_steps", 5),
        check_val_every_n_epoch=config_trainer.get("check_val_every_n_epoch", 1),
        benchmark=config_trainer.get("benchmark", False),
        callbacks=callbacks_list,  # Use the collected list of callbacks
    )

    # Compile the model
    # model = torch.compile(
    #     model,
    #     # fullgraph=False,
    #     # dynamic=True,
    # )

    # Train the model
    start = time.time()
    # Pass the datamodule to trainer.fit - the trainer will call its setup methods
    # Remove the model's internal setup calls here, as the trainer uses the DataModule

    # Start training
    # Use trainer.fit with the datamodule and the resume_from_checkpoint_path
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint_path,  # Use the path from config or None
    )

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save the model (optional, can rely on checkpointing)
    # Use a different name or remove if you rely entirely on Lightning checkpointing
    # model.save_to("base_multiblock_model.pt")
    # print("Base model saved as 'base_multiblock_model.pt'.")

    # Print final losses if needed - You can get these from the logger or LossHistory callback if kept
    print(
        f"Final Training Losses: {loss_history.train_losses[-1] if loss_history.train_losses else 'N/A'}"
    )
    print(
        f"Final Validation Losses: {loss_history.val_losses[-1] if loss_history.val_losses else 'N/A'}"
    )

    # Perform final test evaluation run if test data is available
    # This will use the DataModule's test_dataloader and setup('test') will be called by the trainer
    trainer.test(model, datamodule=data_module)
