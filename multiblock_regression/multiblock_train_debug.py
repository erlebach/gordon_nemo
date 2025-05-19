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

# ----------------------------------------------------------------------
# Remove adapter related classes
# ----------------------------------------------------------------------

# class MultiBlockRegressorModel(ModelPT, adapter_mixins.AdapterModelPTMixin):
class MultiBlockRegressorModel(ModelPT):
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
        # Replaced MultiBlockRegressor with a simple Linear layer
        self.regressor = nn.Linear(cfg.arch.input_dim, cfg.arch.output_dim)

        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Setup dataloaders if configs are available
        if hasattr(cfg, "model"):
            if hasattr(cfg.model, "train_ds"):
                self.setup_training_data(cfg.model.train_ds)
            if hasattr(cfg.model, "validation_ds"):
                self.setup_validation_data(cfg.model.validation_ds)
            if hasattr(cfg.model, "test_ds"):
                self.setup_test_data(cfg.model.test_ds)

    def forward(self, x):
        # Modified forward to use the simple linear layer
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        # Existing minimal implementation
        x, y = batch
        y_hat = self(x=x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Existing minimal implementation
        x, y = batch
        y_hat = self(x=x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Existing minimal implementation
        x, y = batch
        y_hat = self(x=x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Existing minimal implementation
        """Configures the optimizer.

        Returns:
            The configured optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.get("weight_decay", 0.0),
        )

        return optimizer

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        # Existing implementation using _get_dataloader_from_config
        """Setup training data loader.

        Args:
            train_data_config: Configuration for training data
        """
        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        # Existing implementation using _get_dataloader_from_config
        """Setup validation data loader.

        Args:
            val_data_config: Configuration for validation data
        """
        self._validation_dl = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        # Existing implementation using _get_dataloader_from_config
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
        # Ensure numpy is imported locally if needed, or rely on top-level import
        # import numpy as np # Redundant if imported at top

        # Load data from numpy file - Assuming config has 'file_path', 'batch_size', 'num_workers', 'pin_memory'
        # Added checks for config attributes
        if not hasattr(config, 'file_path') or not hasattr(config, 'batch_size'):
             logging.warning("Dataloader config missing file_path or batch_size. Returning None.")
             return None

        try:
            data = np.load(config.file_path)
            x = data["x"]
            y = data["y"]
        except FileNotFoundError:
            logging.warning(f"Data file not found at {config.file_path}. Returning None.")
            return None
        except Exception as e:
            logging.warning(f"Error loading data from {config.file_path}: {e}. Returning None.")
            return None

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

    @property
    def adapter_module_names(self) -> List[str]:
        # This property was likely related to adapters, but might still be needed by the mixin.
        # Returning a minimal list or adapting based on simple linear layer.
        # Given the linear layer, this property doesn't make semantic sense anymore.
        # Let's remove it and see if the mixin complains or if the error changes.
        pass

    @property
    def default_adapter_module_name(self) -> str:
        # REMOVED default_adapter_module_name property
        pass

    def check_valid_model_with_adapter_support_(self):
        # Existing minimal implementation
        """Check if the model supports adapters."""
        pass

    @classmethod
    def list_available_models(cls):
        # Existing minimal implementation
        """
        This method is required by ModelPT and should return a list of
        pretrained model weights available for this class.

        Returns:
            List of pretrained models available for this class
        """
        return []

    def train_dataloader(self):
        # Existing implementation with better error handling
        """Return the training dataloader with better error handling."""
        if not hasattr(self, "_train_dl") or self._train_dl is None:
             # Added check for train_ds config existence before calling setup
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "train_ds"):
                print(
                    f"Attempting to set up training dataloader from config..."
                )
                self.setup_training_data(self.cfg.model.train_ds)
            else:
                logging.warning("No 'train_ds' configuration found in model config. Cannot set up training dataloader.")

        if self._train_dl is None:
             logging.warning("Training dataloader is None.")
             return None

        return self._train_dl

    def val_dataloader(self):
        # Existing implementation with better error handling
        """Return the validation dataloader with better error handling."""
        if not hasattr(self, "_validation_dl") or self._validation_dl is None:
            # Added check for validation_ds config existence before calling setup
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "validation_ds"):
                print(
                    f"Attempting to set up validation dataloader from config..."
                )
                self.setup_validation_data(self.cfg.model.validation_ds)
            else:
                logging.warning("No 'validation_ds' configuration found in model config. Cannot set up validation dataloader.")

        if self._validation_dl is None:
             logging.warning("Validation dataloader is None.")
             return None

        return self._validation_dl

    def test_dataloader(self):
        # Existing implementation with better error handling
        """Return the test dataloader with better error handling."""
        if not hasattr(self, "_test_dl") or self._test_dl is None:
            # Added check for test_ds config existence before calling setup
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "test_ds"):
                print(
                    f"Attempting to set up test dataloader from config..."
                )
                self.setup_test_data(self.cfg.model.test_ds)
            else:
                logging.warning("No 'test_ds' configuration found in model config. Cannot set up test dataloader.")

        if self._test_dl is None:
            logging.warning("Test dataloader is None.")
            return None

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
        # Existing implementation
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
    # Removed data preparation call
    # import os
    # if not os.path.exists("data/sine_train.npz"):
    #     logging.info("Data files not found. Creating datasets...")
    #     prepare_data()


    # Set up trainer with Lightning
    trainer = pl.Trainer(**cfg.trainer)


    # Create model
    # This uses the simplified MultiBlockRegressorModel
    model = MultiBlockRegressorModel(cfg=cfg, trainer=trainer)

    # Training
    logging.info("Starting training...")
    # The dataloader methods will be called by trainer.fit, but they might return None
    # because we haven't set up the data preparation or provided a data file in the config
    # Trainer.fit might raise an error if train_dataloader returns None.
    # We'll address data loading if needed after checking the type error.
    trainer.fit(model)
    logging.info("Training completed.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
