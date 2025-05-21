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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import os
# Remove CSVLogger import
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger # Ensure TensorBoardLogger is also imported
from lightning.pytorch.loggers import TensorBoardLogger # Keep TensorBoardLogger import
from lightning.pytorch.loggers import CSVLogger # Keep CSVLogger import

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

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiBlockRegressorModel.

        Args:
            x: A batch of data from the dataloader.
               Expected to be a list of Tensors [input_tensor, label_tensor]
               or just the input_tensor if labels are not included in the batch
               during prediction or other stages.

        Returns:
            The output of the regressor.
        """
        # Check if the input 'x' is a list (common when dataloader returns features and labels)
        if isinstance(x, list):
            # Extract the input tensor, assuming it's the first element
            input_tensor = x[0]
        else:
            # Assuming 'x' itself is the input tensor
            input_tensor = x

        return self.regressor(input_tensor)

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

    # Exp manager - ONLY use it to get the experiment directory
    # Do NOT pass the trainer to exp_manager here if we are manually creating loggers
    # exp_dir = exp_manager(trainer, cfg.get("exp_manager", None)) # REMOVE this line or comment out
    # Instead, we'll get the log directory based on exp_manager's logic
    # Simulate exp_manager's directory creation to get the log_dir
    exp_cfg = cfg.get("exp_manager", OmegaConf.create({})) # Get exp_manager config safely
    base_log_dir = exp_cfg.get("exp_dir", "./nemo_experiments")
    exp_name = exp_cfg.get("name", cfg.name) # Use model name as default if exp_manager name is null
    version_value = os.getpid() if exp_cfg.get("use_datetime_version", True) else "manual_version" # Provide a default string if not using datetime version
    log_dir = os.path.join(base_log_dir, exp_name, str(version_value))
    os.makedirs(log_dir, exist_ok=True) # Create the directory

    print(f"DEBUG: Manually determined log_dir: {log_dir}")


    # Set up loggers manually
    tensorboard_logger = TensorBoardLogger(save_dir=base_log_dir, name=exp_name, version=version_value)
    csv_logger = CSVLogger(save_dir=base_log_dir, name=exp_name, version=version_value)
    loggers = [tensorboard_logger, csv_logger] # Create a list of loggers

    # Set up trainer with Lightning, passing the list of loggers
    trainer_params = OmegaConf.to_container(cfg.trainer, resolve=True)
    # Pass the list of loggers to the trainer
    trainer_params['logger'] = loggers
    trainer = pl.Trainer(**trainer_params)

    # Debug prints to check the logger after manual setup (keep these)
    print("-" * 20)
    print("DEBUG: Checking trainer.logger after manual setup")
    if trainer.logger:
        print(f"Trainer logger: {trainer.logger}")
        if isinstance(trainer.logger, list):
            print(f"Trainer logger is a list of {len(trainer.logger)} loggers.")
            for i, logger_instance in enumerate(trainer.logger):
                print(f"  Logger {i}: {type(logger_instance)}")
                if hasattr(logger_instance, 'log_dir'):
                     print(f"    Logger {i} log_dir: {logger_instance.log_dir}")
        else:
            print(f"Trainer logger is a single instance: {type(trainer.logger)}")
            if hasattr(trainer.logger, 'log_dir'):
                 print(f"  Logger log_dir: {trainer.logger.log_dir}")
    else:
        print("No logger attached to trainer.")
    print("-" * 20)


    # Training
    logging.info("Starting training...")
    trainer.fit(model)
    logging.info("Training completed.")

    # ---------------------
    # Post-training analysis (will now use the trainer with manually set loggers)
    # ---------------------
    post_train_analysis(model, trainer)


def post_train_analysis(model: MultiBlockRegressorModel, trainer: pl.Trainer):
    """
    Performs post-training analysis including evaluating on validation set
    and plotting predicted vs actual values, and plotting loss curves.

    Args:
        model: The trained MultiBlockRegressorModel instance.
        trainer: The PyTorch Lightning Trainer instance.
    """
    logging.info("Starting post-training analysis...")

    # 1. Run evaluation on the validation dataset
    logging.info("Evaluating on validation dataset...")
    # Ensure val_dataloader is available before validating
    val_dataloader = model.val_dataloader()
    if val_dataloader:
        # Use the trainer object configured by exp_manager
        trainer.validate(model, dataloaders=val_dataloader)
        logging.info("Validation evaluation complete.")
    else:
        logging.warning("Validation dataloader not available. Skipping validation evaluation.")


    # 2. Get predictions for plotting
    logging.info("Generating predictions for training and validation data...")

    # Get predictions for validation data
    val_dataloader = model.val_dataloader() # Get it again in case setup was deferred
    if val_dataloader:
        # Use the trainer object configured by exp_manager
        val_predictions = trainer.predict(model, dataloaders=val_dataloader)
        # Flatten the list of lists if necessary (depends on model output structure)
        # Add a check to ensure val_predictions is not empty before concatenating
        if val_predictions and all(len(p) > 0 for p in val_predictions):
            val_preds_flat = torch.cat(val_predictions).cpu().numpy()

            # Get actual values for validation data.
            val_actuals = []
            # Iterate the dataloader to get labels. Assumes batch is [input, label].
            for batch in val_dataloader:
                 # Assuming the second element in the batch is the label
                 val_actuals.append(batch[1].cpu()) # Move to CPU before appending
            val_actuals_flat = torch.cat(val_actuals).numpy()
        else:
            logging.warning("No validation predictions generated. Skipping validation plot.")
            val_preds_flat = None # Set to None to skip plotting
            val_actuals_flat = None
    else:
        logging.warning("Validation dataloader not available. Skipping validation prediction.")
        val_preds_flat = None # Set to None to skip plotting
        val_actuals_flat = None


    # Get predictions for training data
    train_dataloader_original = model.train_dataloader() # Get it again
    if train_dataloader_original:
        # Create a non-shuffling dataloader for prediction if the original was shuffled
        if hasattr(train_dataloader_original, 'shuffle') and train_dataloader_original.shuffle:
            logging.info("Original training dataloader is shuffled. Creating a non-shuffling dataloader for prediction.")
            train_dataloader_predict = DataLoader(
                dataset=train_dataloader_original.dataset,
                batch_size=train_dataloader_original.batch_size,
                shuffle=False, # Explicitly set shuffle to False
                num_workers=train_dataloader_original.num_workers, # Keep other parameters
                pin_memory=train_dataloader_original.pin_memory,
                drop_last=train_dataloader_original.drop_last,
                timeout=train_dataloader_original.timeout,
                worker_init_fn=train_dataloader_original.worker_init_fn,
                # Add other DataLoader parameters if necessary
            )
        else:
            train_dataloader_predict = train_dataloader_original # Use original if not shuffled

        # Use the trainer object configured by exp_manager
        train_predictions = trainer.predict(model, dataloaders=train_dataloader_predict)
        # Add a check to ensure train_predictions is not empty before concatenating
        if train_predictions and all(len(p) > 0 for p in train_predictions):
            train_preds_flat = torch.cat(train_predictions).cpu().numpy()

            # Get actual values for training data using the non-shuffling dataloader
            train_actuals = []
            for batch in train_dataloader_predict:
                # Assuming the second element in the batch is the label
                 train_actuals.append(batch[1].cpu()) # Move to CPU before appending
            train_actuals_flat = torch.cat(train_actuals).numpy()
        else:
            logging.warning("No training predictions generated. Skipping training plot.")
            train_preds_flat = None # Set to None to skip plotting
            train_actuals_flat = None
    else:
        logging.warning("Training dataloader not available. Skipping training prediction.")
        train_preds_flat = None # Set to None to skip plotting
        train_actuals_flat = None

    logging.info("Prediction generation complete.")


    # 3. Plotting Predicted vs Actual
    logging.info("Plotting predicted vs actual values...")
    # Only create the figure if we have data to plot
    if train_preds_flat is not None or val_preds_flat is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot for Training Data
        if train_preds_flat is not None:
            axes[0].scatter(train_actuals_flat, train_preds_flat, alpha=0.5)
            axes[0].set_title('Training Data: Predicted vs Actual')
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            # Add a line for perfect prediction
            min_train = min(train_actuals_flat.min(), train_preds_flat.min())
            max_train = max(train_actuals_flat.max(), train_preds_flat.max())
            axes[0].plot([min_train, max_train], [min_train, max_train], 'r--')
            axes[0].set_aspect('equal', adjustable='box') # Make scales equal for better visual comparison
        else:
            axes[0].set_title('Training Data: Predicted vs Actual (No Data)')
            axes[0].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)


        # Plot for Validation Data
        if val_preds_flat is not None:
            axes[1].scatter(val_actuals_flat, val_preds_flat, alpha=0.5, color='orange')
            axes[1].set_title('Validation Data: Predicted vs Actual')
            axes[1].set_xlabel('Actual Values')
            axes[1].set_ylabel('Predicted Values')
            # Add a line for perfect prediction
            min_val = min(val_actuals_flat.min(), val_preds_flat.min())
            max_val = max(val_actuals_flat.max(), val_preds_flat.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[1].set_aspect('equal', adjustable='box') # Make scales equal for better visual comparison
        else:
            axes[1].set_title('Validation Data: Predicted vs Actual (No Data)')
            axes[1].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


        plt.tight_layout()
        # plt.show() # Don't show yet, add loss plot
        plt.savefig("pred_vs_actual.jpg")
        print("pred_vs_actual.jpg saved")
    else:
        logging.warning("No data available for plotting predicted vs actual.")


    # 4. Plotting Loss over Epochs
    logging.info("==> Plotting loss over epochs...")
    # Use the experiment directory returned by exp_manager
    log_dir = trainer.log_dir # This gets the logger's directory from the trainer configured by exp_manager
    print("==> log_dir= ", log_dir)
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    print("==> metrics_path= ", metrics_path)

    if os.path.exists(metrics_path):
        try:
            metrics_df = pd.read_csv(metrics_path)

            # Filter out rows where 'epoch' is NaN (these are often step-level metrics)
            epoch_metrics = metrics_df.dropna(subset=['epoch'])

            # Ensure 'epoch' is integer for plotting
            epoch_metrics['epoch'] = epoch_metrics['epoch'].astype(int)

            # Group by epoch to handle multiple log entries per epoch (e.g., step losses)
            # and get the last logged value per epoch for train_loss and val_loss
            # Assumes the epoch-level loss is logged last within an epoch
            # Alternatively, you could average if that makes sense for your logging
            epoch_metrics = epoch_metrics.groupby('epoch').tail(1).reset_index(drop=True)


            if 'train_loss_epoch' in epoch_metrics.columns or 'val_loss_epoch' in epoch_metrics.columns:
                 # Create a new figure for the loss plot, or add to existing if desired
                 # For clarity, let's create a new figure
                 plt.figure(figsize=(10, 5))

                 if 'train_loss_epoch' in epoch_metrics.columns:
                    plt.plot(epoch_metrics['epoch'], epoch_metrics['train_loss_epoch'], label='Training Loss')
                 if 'val_loss_epoch' in epoch_metrics.columns:
                     # Only plot validation loss if available
                     # Ensure there are non-NaN values for val_loss_epoch
                     val_loss_data = epoch_metrics.dropna(subset=['val_loss_epoch'])
                     if not val_loss_data.empty:
                         plt.plot(val_loss_data['epoch'], val_loss_data['val_loss_epoch'], label='Validation Loss')
                     else:
                         logging.warning("No non-NaN 'val_loss_epoch' found in metrics.csv.")


                 plt.xlabel('Epoch')
                 plt.ylabel('Loss')
                 plt.title('Loss over Epochs')
                 plt.legend()
                 plt.grid(True)
                 # plt.show() # Don't show yet, show all plots together
            else:
                logging.warning("'train_loss_epoch' or 'val_loss_epoch' not found in metrics.csv.")
        except Exception as e:
            logging.warning(f"Error plotting loss from metrics.csv: {e}")
    else:
        logging.warning(f"metrics.csv not found at {metrics_path}. Cannot plot loss over epochs.")


    # Show all generated plots
    plt.show()
    logging.info("All plotting complete.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()