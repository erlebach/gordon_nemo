import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.core import ModelPT
from nemo.core.classes import adapter_mixins
from nemo.core.config import hydra_runner
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
class MinimalModelPT(ModelPT, adapter_mixins.AdapterModelPTMixin):
    # Expect cfg to contain only model-specific parameters (like optim, arch)
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # Pass the model-specific config and trainer to ModelPT's init
        super().__init__(cfg=cfg, trainer=trainer)

        # Minimal model layer - Access arch config from self.cfg
        # Using input_dim/output_dim from cfg.arch as in multiblock_train_debug
        input_dim = cfg.arch.get("input_dim", 1) # Get from config or default
        output_dim = cfg.arch.get("output_dim", 1) # Get from config or default
        self.linear = nn.Linear(input_dim, output_dim)
        self.criterion = nn.MSELoss()

        # Initialize dataloader attributes
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None

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
    # Modified to use _get_dataloader_from_config and store dataloaders, mirroring debug code
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict, None]):
        if train_data_config is not None:
            logging.info("Setting up training dataloader from config.")
            self._train_dl = self._get_dataloader_from_config(
                train_data_config, shuffle=True
            )
            if self._train_dl is None:
                 logging.warning("Failed to create training dataloader from config.")
            else:
                 logging.info("Training dataloader set.")

        else:
            self._train_dl = None
            logging.info("setup_training_data called with None config.")

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict, None]):
        if val_data_config is not None:
            logging.info("Setting up validation dataloader from config.")
            self._validation_dl = self._get_dataloader_from_config(
                val_data_config, shuffle=False
            )
            if self._validation_dl is None:
                 logging.warning("Failed to create validation dataloader from config.")
            else:
                 logging.info("Validation dataloader set.")

        else:
            self._validation_dl = None
            logging.info("setup_validation_data called with None config.")

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict, None]):
        if test_data_config is not None:
            logging.info("Setting up test dataloader from config.")
            self._test_dl = self._get_dataloader_from_config(
                test_data_config, shuffle=False
            )
            if self._test_dl is None:
                 logging.warning("Failed to create test dataloader from config.")
            else:
                 logging.info("Test dataloader set.")
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

    # Added _get_dataloader_from_config helper method with implementation, mirroring debug code
    def _get_dataloader_from_config(self, config: Union[DictConfig, Dict], shuffle: bool) -> DataLoader:
        # Implementation of _get_dataloader_from_config method
        # This is a placeholder and should be implemented based on your specific requirements
        return None  # Placeholder return, actual implementation needed

    # --- Adapter related methods (minimal placeholders due to mixin inheritance) ---
    # These are needed because MinimalModelPT inherits from AdapterModelPTMixin, mirroring debug code
    @property
    def adapter_module_names(self) -> List[str]:
         return [] # Minimal implementation

    @property
    def default_adapter_module_name(self) -> str:
         return "" # Minimal implementation

    def check_valid_model_with_adapter_support_(self):
         pass # Minimal implementation

    def add_adapter(self, name: str, cfg: Union[DictConfig, Dict]):
         logging.warning("add_adapter called but not implemented.")
         pass # Minimal implementation

    def get_enabled_adapters(self):
         return [] # Minimal implementation

    def is_adapter_available(self) -> bool:
         return False # Minimal implementation

    def set_enabled_adapters(self, name=None, enabled=True):
         logging.warning("set_enabled_adapters called but not implemented.")
         pass # Minimal implementation

    @classmethod
    def restore_from(
        cls,
        restore_path,
        trainer=None,
        override_config_path=None,
        map_location=None,
        strict=True,
    ):
        # Minimal implementation, just calls super, mirroring debug code
        return super().restore_from(
            restore_path=restore_path,
            trainer=trainer,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=strict,
        )


# 3. Main execution block using @hydra_runner
@hydra_runner(config_path=".", config_name="minimal_config.yaml")
def main(cfg: DictConfig) -> None:
    print("Hydra main function entered.")
    logging.info("Testing MinimalModelPT with PyTorch Lightning Trainer...")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Instantiate trainer using config loaded by hydra_runner
    trainer = pl.Trainer(**cfg.trainer)

    # Add exp_manager call (optional, can add later if needed)
    # exp_manager(trainer, cfg.get("exp_manager", None))

    try:
        logging.info("Instantiating MinimalModelPT...")
        # Instantiate the model - pass the relevant model config (cfg.model) and trainer
        # Matching the working debug file's direct instantiation pattern
        # Assuming model config is under cfg.model as in the debug file
        model = MinimalModelPT(cfg=cfg.model, trainer=trainer)
        logging.info("MinimalModelPT instantiated successfully.")

        # Removed manual set_trainer call - trainer is passed in constructor
        # Removed manual setup_data calls - trainer.fit/test will call them if dataloaders are None

        # Now call trainer.fit.
        logging.info("Calling trainer.fit...")
        # Trainer.fit will call setup methods and dataloader methods automatically
        # Pass dataset configs if available in the main cfg, otherwise the setup methods will use model.train_ds etc.
        # Rely on the setup methods being called automatically by trainer.fit/test
        trainer.fit(model)
        logging.info("trainer.fit completed.") # Changed message as we hope it completes now

        # Add test call if config is available
        if hasattr(cfg, "test_ds"):
             logging.info("\nCalling trainer.test...")
             trainer.test(model)
             logging.info("trainer.test completed.")

    except Exception as e: # Catch generic Exception for broader error capture
        logging.error(f"\nCaught an error: {e}")
        import traceback

        traceback.print_exc()

    logging.info("\nTest finished.")


# This block will be the entry point when running the script directly
if __name__ == "__main__":
    print("Executing __main__ block.")
    main() # Call the hydra_runner decorated main function
