import argparse  # Import the argparse module
import os

import lightning.pytorch as pl
import numpy as np
import torch
from exp_lightning.multiblock_regressor_nemo import (
    # MultiBlockRegressor,
    MultiBlockRegressorNeMo,
)
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


# Define a custom callback to save predictions and targets
class SaveValidationOutputsCallback(Callback):
    """
    A callback to collect predictions and targets during validation
    and save them to a file at the end of the epoch.
    """

    def __init__(self, output_filepath: str):
        super().__init__()
        self.output_filepath = output_filepath
        self.predictions = []
        self.targets = []

    def on_validation_epoch_start(self, trainer, pl_module):
        """Initialize lists at the start of the validation epoch."""
        self.predictions = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect predictions and targets from each batch."""
        # Get inputs (x) and targets (y) from the batch
        x, y = batch

        # Get predictions (y_hat) by running the model with the input batch
        # Ensure the model is in evaluation mode and on the correct device
        # .eval() and .to(device) are handled by trainer.validate internally,
        # but calling the module directly might need .eval() if not already set.
        # Let's rely on trainer.validate to manage model state for simplicity.
        with torch.no_grad():  # Ensure no gradients are calculated during this call
            y_hat = pl_module(x=x)

        # Ensure tensors are on CPU and detached before converting to numpy
        self.predictions.append(y_hat.detach().cpu())
        self.targets.append(y.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Concatenate collected tensors and save them."""
        if self.predictions and self.targets:
            # Concatenate all batch tensors
            all_predictions = torch.cat(self.predictions).numpy()
            all_targets = torch.cat(self.targets).numpy()

            # Save data to an .npz file
            # np.savez allows saving multiple arrays in a single file
            np.savez(
                self.output_filepath, predictions=all_predictions, targets=all_targets
            )

            print(f"Saved validation predictions and targets to {self.output_filepath}")
        else:
            print("No validation data collected to save.")

        # Clear lists for the next epoch (though typically only one val epoch in this script)
        self.predictions = []
        self.targets = []


def evaluate_saved_model(model_path: str, data_path: str, output_save_path: str):
    """Loads a saved model and runs validation on specified data.

    Args:
        model_path: Path to the saved .nemo model file.
        data_path: Path to the .npz file containing validation data (x and y).
        output_save_path: Path to save the predicted and actual validation data.
    """
    print(f"Loading model config from: {model_path}")
    # Load the config using restore_from, but don't instantiate yet
    # The config loaded here should be the original config used for saving
    loaded_config = MultiBlockRegressorNeMo.restore_from(
        restore_path=model_path, trainer=None, return_config=True
    )
    print("Model config loaded successfully.")
    print("loaded_config = ", OmegaConf.to_yaml(loaded_config))

    print("Instantiating model from loaded config...")
    # Manually instantiate the model using the loaded config
    # Pass the loaded_config directly to your model's __init__
    # Your __init__ should handle the instantiation of sub-modules like 'arch'
    model = MultiBlockRegressorNeMo(cfg=loaded_config, trainer=None)
    print("Model instantiated successfully.")

    # Load the state dict separately if needed, though restore_from(return_config=True) might handle it
    # Need to confirm if restore_from(return_config=True) also loads state_dict or if it's implicit
    # If not implicit, you might need:
    # model.load_state_dict(torch.load(model_path)['state_dict']) # This path might vary

    print(f"Loading data from: {data_path}")
    # --- Data Loading Logic (adapted from _get_dataloader_from_config) ---
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        data = np.load(data_path)
        x = data["x"]
        y = data["y"]

        # Ensure tensors have shape (N, 1) if input_dim or output_dim is 1
        # Access model.regressor for input/output dims AFTER model is instantiated
        if model.regressor.input_dim == 1 and x.ndim == 1:
            x = x[:, None]
        if model.regressor.output_dim == 1 and y.ndim == 1:
            y = y[:, None]

        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

        batch_size = 64  # Example batch size, make it configurable if needed
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle evaluation data
            num_workers=0,  # Or configure as needed
            pin_memory=False,  # Or configure as needed
        )
        print(f"DataLoader created successfully for {data_path}.")

    except Exception as e:
        print(f"Error creating dataloader from {data_path}: {e}")
        return  # Exit if data loading fails
    # --- End Data Loading Logic ---

    # Instantiate the custom callback
    save_callback = SaveValidationOutputsCallback(output_filepath=output_save_path)

    print("Setting up trainer for validation...")
    # Instantiate a minimal trainer for evaluation
    # You can configure accelerators, devices etc. as needed
    trainer = pl.Trainer(
        logger=False,  # No need for extensive logging during evaluation
        accelerator="auto",  # Use available accelerator
        devices=1,  # Use 1 device
        callbacks=[save_callback],  # Pass the callback to the trainer
        # Add any other necessary trainer args like precision, etc.
    )
    print("Trainer setup complete.")

    print("Running validation...")
    # Run the validation loop
    # The trainer.validate method takes the model and the dataloader(s)
    # It will call the model's validation_step and validation_epoch_end hooks
    validation_results = trainer.validate(model, dataloaders=dataloader)
    print("Validation completed.")

    print("Validation Results:", validation_results)

    # The predictions and targets are saved by the callback in on_validation_epoch_end


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate a saved NeMo model.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved .nemo model file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the .npz file containing validation data (x and y).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the predicted and actual validation data (.npz file).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Ensure the output directory exists if it's nested
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Call the evaluation function with parsed arguments
    evaluate_saved_model(args.model_path, args.data_path, args.output_path)

    # Optional: Load and check the saved file
    # try:
    #     loaded_data = np.load(args.output_path)
    #     loaded_preds = loaded_data['predictions']
    #     loaded_targets = loaded_data['targets']
    #     print(f"\nSuccessfully loaded saved data from {args.output_path}")
    #     print(f"Loaded predictions shape: {loaded_preds.shape}")
    #     print(f"Loaded targets shape: {loaded_targets.shape}")
    # except Exception as e:
    #     print(f"Error loading saved data: {e}")
