"""
A version of multiblock_train.py with all classes in a single file. 
This will allow a series of simplifications to try and control
the error produced, which has to do with how the constructor is 
initialized (with both cfg and trainer). 
"""
from typing import Any, Dict, List, Optional, Tuple, Union
from nemo.core import ModelPT, adapter_mixins
from torch import nn

# multiblock_train.py
import lightning.pytorch as pl
import numpy as np
import torch
# from multiblock_adapter import MultiBlockRegressorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
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
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
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
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "test_ds"):
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

@hydra_runner(config_path=".", config_name="multiblock_debug_config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if data needs to be prepared
    import os


    # Set up trainer with Lightning
    # GE:  What are arguments of pl.Trainer?
    trainer = pl.Trainer(**cfg.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))

    # Create model
    # GE: look into trainer in base model
    # this has information about the adapter. WHY? 
    model = MultiBlockRegressorModel(cfg=cfg, trainer=trainer)

    # Training
    logging.info("Starting training...")
    # GE lightning model
    trainer.fit(model)
    logging.info("Training completed.")

    # Testing
    if hasattr(cfg, "test_ds") and cfg.test_ds.file_path is not None:
        # GE: Should reach this point
        logging.info("Running testing...")
        # GE: Lightning
        trainer.test(model)

    # Save the model if a path is specified
    if (
        hasattr(cfg, "io")
        and hasattr(cfg.io, "nemo_path")
        and cfg.io.nemo_path is not None
    ):
        model.save_to(cfg.io.nemo_path)
        logging.info(f"Model saved to {cfg.io.nemo_path}")
    elif (
        hasattr(cfg, "model")
        and hasattr(cfg.model, "nemo_path")
        and cfg.model.nemo_path is not None
    ):
        model.save_to(cfg.model.nemo_path)
        logging.info(f"Model saved to {cfg.model.nemo_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
