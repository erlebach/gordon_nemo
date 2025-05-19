import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.core import ModelPT
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Add basic logging configuration for visibility
logging.basicConfig(level=logging.INFO)


# 1. Define a minimal dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 1)  # Dummy input
        self.labels = torch.randn(size, 1)  # Dummy target

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 2. Define a minimal ModelPT subclass with required abstract methods implemented
class MinimalModelPT(ModelPT):
    # Expect cfg to contain only model-specific parameters (like optim)
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # Pass the model-specific config and trainer to ModelPT's init
        # Pass trainer as provided (will be None initially in the test)
        super().__init__(cfg=cfg, trainer=trainer)

        # Minimal model layer - Access optim config from self.cfg
        self.linear = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward pass
        return self.linear(x)

    # --- LightningModule required methods ---
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)  # Log loss
        return loss

    def configure_optimizers(self):
        # Access optim config from self.cfg
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        return optimizer

    # --- ModelPT data setup methods (Required abstract methods) ---
    # These methods receive the *full* experiment config sections (e.g., cfg.train_ds)
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        if train_data_config is not None:
            dataset = DummyDataset(size=train_data_config.dataset_size)
            self._train_dl = DataLoader(
                dataset, batch_size=train_data_config.batch_size
            )
            logging.info("setup_training_data called and dataloader set.")
        else:
            self._train_dl = None
            logging.info("setup_training_data called with None config.")

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        if val_data_config is not None:
            dataset = DummyDataset(size=val_data_config.dataset_size)
            self._validation_dl = DataLoader(
                dataset, batch_size=val_data_config.batch_size
            )
            logging.info("setup_validation_data called and dataloader set.")
        else:
            self._validation_dl = None
            logging.info("setup_validation_data called with None config.")

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        if test_data_config is not None:
            dataset = DummyDataset(size=test_data_config.dataset_size)
            self._test_dl = DataLoader(dataset, batch_size=test_data_config.batch_size)
            logging.info("setup_test_data called and dataloader set.")
        else:
            self._test_dl = None
            logging.info("setup_test_data called with None config.")

    # --- LightningModule data hooks (delegates to _train_dl etc) ---
    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._validation_dl

    def test_dataloader(self):
        return self._test_dl

    # --- Other required abstract methods from ModelPT ---
    @classmethod
    def list_available_models(cls) -> Optional[List[Tuple[str, str]]]:
        return None


# 3. Main execution block to test
if __name__ == "__main__":
    logging.info("Testing MinimalModelPT with PyTorch Lightning Trainer...")

    # Create the full experiment config structure
    full_experiment_cfg = OmegaConf.create(
        {
            "model": {  # Model specific config nested under 'model' key
                "optim": {"lr": 0.01},
                # Other model params would go here (e.g., arch)
            },
            "train_ds": {"dataset_size": 100, "batch_size": 10},
            "validation_ds": {"dataset_size": 50, "batch_size": 10},
            "test_ds": {"dataset_size": 50, "batch_size": 10},
            # Other experiment level configs (trainer, exp_manager) would be here
        }
    )

    # Create a minimal trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Instantiate the ModelPT subclass with trainer=None
    try:
        logging.info("Instantiating MinimalModelPT with trainer=None...")
        model = MinimalModelPT(cfg=full_experiment_cfg.model, trainer=None)
        logging.info("MinimalModelPT instantiated successfully.")

        # Set the trainer explicitly after instantiation
        logging.info("Setting trainer using model.set_trainer...")
        model.set_trainer(trainer)
        logging.info("Trainer set successfully.")

        # Now call trainer.fit/test. Pass config objects.
        # Trainer.fit/test will call the setup_data methods based on these configs.
        logging.info("Calling trainer.fit...")
        trainer.fit(
            model,
            train_dataloaders=full_experiment_cfg.train_ds,
            val_dataloaders=full_experiment_cfg.validation_ds,
        )
        logging.info("trainer.fit completed successfully.")

        logging.info("\nCalling trainer.test...")
        trainer.test(model, dataloaders=full_experiment_cfg.test_ds)
        logging.info("trainer.test completed successfully.")

    except TypeError as e:
        logging.error(f"\nCaught a TypeError: {e}")
        logging.error("This suggests an issue with type checking.")
        import traceback

        traceback.print_exc()
    except ValueError as e:
        logging.error(f"\nCaught a ValueError: {e}")
        logging.error("This suggests a configuration structure issue.")
        import traceback

        traceback.print_exc()
    except Exception as e:
        logging.error(f"\nCaught an unexpected error: {e}")
        import traceback

        traceback.print_exc()

    logging.info("\nTest finished.")
