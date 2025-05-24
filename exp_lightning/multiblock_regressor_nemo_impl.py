"""NeMo-compatible implementation for MultiBlock Regressor callbacks and utilities.

This module provides NeMo-compatible callbacks and utilities for the MultiBlock Regressor,
adapted from the Lightning implementation to work with NeMo's framework.
"""

import logging
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from nemo.utils import logging as nemo_logging
from omegaconf import DictConfig, OmegaConf


class NeMoLossHistory(Callback):
    """NeMo-compatible callback to store training and validation losses for plotting.

    This callback is compatible with NeMo's training framework and stores loss values
    that can be accessed after training for analysis and plotting.
    """

    def __init__(self):
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when the train epoch ends.

        Args:
            trainer: The trainer instance.
            pl_module: The LightningModule instance.

        """
        # Get train loss from logged metrics
        train_loss = trainer.callback_metrics.get("train_loss_epoch")
        if train_loss is None:
            train_loss = trainer.callback_metrics.get("train_loss")

        if train_loss is not None:
            if isinstance(train_loss, torch.Tensor):
                self.train_losses.append(train_loss.cpu().item())
            else:
                self.train_losses.append(float(train_loss))

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the validation epoch ends.

        Args:
            trainer: The trainer instance.
            pl_module: The LightningModule instance.
        """
        # Get validation loss from logged metrics
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        if val_loss is None:
            val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is not None:
            if isinstance(val_loss, torch.Tensor):
                self.val_losses.append(val_loss.cpu().item())
            else:
                self.val_losses.append(float(val_loss))


class NeMoPredictionPlotter(Callback):
    """NeMo-compatible callback to plot model predictions on test data periodically.

    This callback generates prediction plots during training and logs them to the
    experiment manager's logger (typically TensorBoard).
    """

    def __init__(
        self,
        test_data_path: str,
        loss_history_callback: NeMoLossHistory,
        plot_interval: int = 2,
    ):
        """Initialize the prediction plotter.

        Args:
            test_data_path: Path to the test data file.
            loss_history_callback: Instance of NeMoLossHistory to get loss values.
            plot_interval: Interval (in epochs) for generating plots.
        """
        super().__init__()
        self.test_data_path = test_data_path
        self.loss_history = loss_history_callback
        self.plot_interval = plot_interval

        # Load test data once during initialization
        try:
            data = np.load(test_data_path)
            self.x_test_plot = data["x"]
            self.y_test_plot = data["y"]
            nemo_logging.info(f"Loaded test data for plotting from {test_data_path}")
        except Exception as e:
            nemo_logging.error(f"Error loading test data for plotting: {e}")
            self.x_test_plot = None
            self.y_test_plot = None

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generate and log prediction plots at specified intervals.

        Args:
            trainer: The trainer instance.
            pl_module: The LightningModule instance.
        """
        current_epoch = trainer.current_epoch

        # Generate plot at specified intervals
        if (
            current_epoch > 0 and (current_epoch + 1) % self.plot_interval == 0
        ) or current_epoch == 0:
            if (
                self.x_test_plot is not None
                and self.loss_history.train_losses
                and self.loss_history.val_losses
            ):
                nemo_logging.info(
                    f"Generating prediction plot for epoch {current_epoch + 1}..."
                )

                # Get the most recent losses
                try:
                    train_loss_val = self.loss_history.train_losses[-1]
                    val_loss_val = self.loss_history.val_losses[-1]
                except IndexError:
                    train_loss_val = float("nan")
                    val_loss_val = float("nan")
                    nemo_logging.warning(
                        f"Loss history lists are empty at epoch {current_epoch + 1}. "
                        "Cannot include losses in plot title."
                    )

                # Generate predictions
                pl_module.eval()
                with torch.no_grad():
                    x_test_tensor_plot = torch.tensor(
                        self.x_test_plot, dtype=torch.float32
                    ).to(pl_module.device)
                    y_pred_plot = pl_module(x=x_test_tensor_plot).cpu().numpy()
                pl_module.train()

                # Prepare data for plotting
                y_test_flat = self.y_test_plot.flatten()
                y_pred_flat = y_pred_plot.flatten()

                # Create comprehensive plot
                fig = self._create_prediction_plot(
                    self.x_test_plot,
                    self.y_test_plot,
                    y_pred_plot,
                    y_test_flat,
                    y_pred_flat,
                    current_epoch,
                    train_loss_val,
                    val_loss_val,
                )

                # Log figure to experiment manager
                self._log_figure_to_experiment_manager(trainer, fig, current_epoch)

                plt.close(fig)  # Free memory

            elif self.x_test_plot is None:
                nemo_logging.info(
                    f"Skipping plot for epoch {current_epoch + 1}: Test data not loaded."
                )

    def _create_prediction_plot(
        self,
        x_test,
        y_test,
        y_pred,
        y_test_flat,
        y_pred_flat,
        current_epoch,
        train_loss_val,
        val_loss_val,
    ):
        """Create a comprehensive prediction plot.

        Args:
            x_test: Test input data.
            y_test: Test target data.
            y_pred: Model predictions.
            y_test_flat: Flattened test targets.
            y_pred_flat: Flattened predictions.
            current_epoch: Current training epoch.
            train_loss_val: Current training loss.
            val_loss_val: Current validation loss.

        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract x values for plotting (handle multi-dimensional input)
        x_plot_for_2d = x_test[:, 0] if x_test.ndim > 1 else x_test

        # Main plot (top left) - x vs y and prediction
        axes[0, 0].plot(
            x_plot_for_2d,
            y_test,
            "bo",
            markersize=3,
            alpha=0.5,
            label="Original Data (Target)",
        )
        axes[0, 0].plot(x_plot_for_2d, y_pred, "r-", label="Model Prediction")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        axes[0, 0].set_title("Predictions vs. Ground Truth")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot (top right) - y_true vs y_pred, colored by X value
        scatter_plot = axes[0, 1].scatter(
            y_test_flat, y_pred_flat, c=x_plot_for_2d, cmap="viridis", alpha=0.6, s=10
        )

        # Add diagonal line
        min_val = min(y_test_flat.min(), y_pred_flat.min())
        max_val = max(y_test_flat.max(), y_pred_flat.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7)
        axes[0, 1].set_xlabel("Actual Values")
        axes[0, 1].set_ylabel("Predicted Values")
        axes[0, 1].set_title("Predictions vs. Actual (Color by X)")
        axes[0, 1].grid(True, alpha=0.3)

        # Add colorbar
        cbar = fig.colorbar(scatter_plot, ax=axes[0, 1])
        cbar.set_label("X Value")

        # Residual plot (bottom left)
        residuals = y_test_flat - y_pred_flat
        sample_indices = np.arange(len(residuals))
        axes[1, 0].scatter(sample_indices, residuals, alpha=0.5, color="purple")
        axes[1, 0].axhline(y=0, color="k", linestyle="--")
        axes[1, 0].set_xlabel("Sample Index")
        axes[1, 0].set_ylabel("Residuals (Actual - Predicted)")
        axes[1, 0].set_title("Residual Plot")
        axes[1, 0].grid(True, alpha=0.3)

        # Error histogram (bottom right)
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color="orange")
        axes[1, 1].set_xlabel("Prediction Error")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Error Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        # Add overall title with epoch and loss information
        overall_title = (
            f"Epoch {current_epoch + 1} | "
            f"Train Loss: {train_loss_val:.4f} | "
            f"Val Loss: {val_loss_val:.4f}"
        )
        fig.suptitle(overall_title, fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        return fig

    def _log_figure_to_experiment_manager(self, trainer, fig, current_epoch):
        """Log the figure to the experiment manager's logger.

        Args:
            trainer: The trainer instance.
            fig: The matplotlib figure to log.
            current_epoch: Current training epoch.
        """
        if trainer and trainer.logger:
            try:
                # Log to TensorBoard if available
                if hasattr(trainer.logger, "experiment"):
                    trainer.logger.experiment.add_figure(
                        "Prediction Plot",
                        fig,
                        global_step=trainer.global_step,
                    )
                    nemo_logging.info(
                        f"Prediction plot logged to TensorBoard for epoch {current_epoch + 1}."
                    )
                else:
                    nemo_logging.warning(
                        f"Logger experiment not available. "
                        f"Skipping plot logging for epoch {current_epoch + 1}."
                    )
            except Exception as e:
                nemo_logging.error(f"Error logging figure: {e}")
        else:
            nemo_logging.warning(
                f"Trainer or logger not available. "
                f"Skipping plot logging for epoch {current_epoch + 1}."
            )


def create_nemo_callbacks(
    cfg: DictConfig,
    log_dir: str,
    test_data_path: str | None,
    base_model_eval_path: str | None = None,
    base_validation_ds_config: DictConfig | dict | None = None,
    freq_checkpoint_params: DictConfig | dict | None = None,
) -> list[Callback]:
    """Create custom NeMo callbacks.

    Args:
        cfg: Hydra configuration object (full config).
        log_dir: The root logging directory for the current experiment run.
        test_data_path: Path to the test data file.
        base_model_eval_path: Optional path to the .npz file with pre-computed base model evaluation data.
        base_validation_ds_config: Optional configuration for the base validation dataset.
        freq_checkpoint_params: Configuration for frequency-based checkpointing.

    Returns:
        A list of custom callbacks.
    """
    callbacks = []

    # Check if LoRA is enabled based on lora_rank in the model config
    # Access lora_rank from cfg.model.adapter, providing defaults for safety
    model_cfg = cfg.get("model", OmegaConf.create({}))
    adapter_cfg = model_cfg.get("adapter", OmegaConf.create({"lora_rank": 0}))
    lora_rank = adapter_cfg.get("lora_rank", 0)

    # Assume plot_interval is configured in cfg.trainer or a dedicated plotting section
    # It's better to put plot_interval under custom_callbacks_config
    plot_interval = cfg.get("custom_callbacks_config", {}).get(
        "plot_interval", 2
    )  # Default to plotting every 2 epochs

    # --- Conditional Plotting Callbacks (only if LoRA is enabled) ---
    if lora_rank > 0:
        logging.info(f"LoRA enabled (rank > 0). Adding plotting callbacks.")
        # --- Plotting for Fine-tuned Validation Data ---\
        # Access validation_ds from cfg.model
        finetune_validation_data_config = model_cfg.get("validation_ds", {})
        if finetune_validation_data_config.get("file_path") and plot_interval > 0:
            finetune_validation_data_path = finetune_validation_data_config.get(
                "file_path"
            )  # Use .get for safety
            finetune_validation_batch_size = finetune_validation_data_config.get(
                "batch_size", 32
            )

            callbacks.append(
                PlottingCallback(
                    validation_data_path=finetune_validation_data_path,
                    validation_batch_size=finetune_validation_batch_size,
                    plot_interval=plot_interval,
                    base_model_eval_path=base_model_eval_path,
                    plot_title_suffix=" (Fine-tuned Validation Data)",
                )
            )

        # --- Optional: Plotting for Base Validation Data at the End of Training ---\
        # This config is now expected under custom_callbacks_config
        base_validation_ds_config_from_cfg = cfg.get("custom_callbacks_config", {}).get(
            "base_validation_ds", {}
        )

        if (
            base_model_eval_path
            and base_validation_ds_config_from_cfg  # Use the config from custom_callbacks_config
            and base_validation_ds_config_from_cfg.get("file_path")
        ):
            base_validation_data_path = base_validation_ds_config_from_cfg.get(
                "file_path"
            )  # Use .get for safety
            base_validation_batch_size = base_validation_ds_config_from_cfg.get(
                "batch_size", 32
            )

            callbacks.append(
                BaseDataComparisonPlottingCallback(
                    base_validation_data_path=base_validation_data_path,
                    base_model_eval_path=base_model_eval_path,
                    base_validation_batch_size=base_validation_batch_size,
                    plot_title_suffix=" (Base Validation Data)",
                )
            )
    else:
        logging.info(f"LoRA disabled (rank is 0). Skipping plotting callbacks.")

    # Add TestingCallback if test data is available
    if test_data_path and Path(test_data_path).exists():
        # Test interval and batch size are read from main config
        test_interval = cfg.get("trainer", {}).get("test_interval", -1)
        test_batch_size = cfg.get("model", {}).get("test_ds", {}).get("batch_size", 32)
        if test_interval > 0:  # Only add if test_interval is positive
            callbacks.append(
                TestingCallback(
                    test_data_path=test_data_path,
                    test_batch_size=test_batch_size,
                    test_interval=test_interval,
                )
            )

    # --- Add Frequency-Based Checkpoint Callback ---\
    if freq_checkpoint_params:
        # The checkpoint directory is a subdirectory of the log_dir set by exp_manager
        checkpoint_dir = os.path.join(
            log_dir, "checkpoints", "freq_checkpoints"
        )  # Added subdirectory name
        # Ensure the directory exists (exp_manager usually creates this, but good practice)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Check if frequency saving is actually configured (either by epoch or steps)
        is_frequency_configured = False
        every_n_epochs = freq_checkpoint_params.get("every_n_epochs")
        every_n_train_steps = freq_checkpoint_params.get("every_n_train_steps")

        if every_n_epochs is not None or every_n_train_steps is not None:
            is_frequency_configured = True

        if is_frequency_configured:
            # Create the ModelCheckpoint instance, passing parameters directly from the config
            # Use default values if not provided in the config
            freq_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=freq_checkpoint_params.get(
                    "filename", "freq_epoch={epoch}-step={step}"
                ),
                every_n_epochs=every_n_epochs,  # Use extracted value
                every_n_train_steps=every_n_train_steps,  # Use extracted value
                save_top_k=freq_checkpoint_params.get(
                    "save_top_k", -1
                ),  # Default to saving all freq checkpoints
                save_last=freq_checkpoint_params.get(
                    "save_last", False
                ),  # Default to not saving last here
                # monitor and mode are not needed for frequency saving with every_n_ criteria
                # save_on_train_epoch_end: freq_checkpoint_params.get("save_on_train_epoch_end", False) # Add if needed
            )
            callbacks.append(freq_checkpoint_callback)
            # Log which frequency is being used
            if every_n_epochs is not None:
                logging.info(
                    f"Added frequency checkpoint callback saving every {every_n_epochs} epochs to {checkpoint_dir}"
                )
            elif every_n_train_steps is not None:
                logging.info(
                    f"Added frequency checkpoint callback saving every {every_n_train_steps} steps to {checkpoint_dir}"
                )

        else:
            logging.warning(
                "freq_checkpoint_params provided but neither every_n_epochs nor every_n_train_steps is specified. "
                "Frequency checkpoint callback not added."
            )

    return callbacks


class PlottingCallback(Callback):
    """Callback for plotting predictions vs actuals during validation (typically on fine-tuned data).

    Plots the predicted values from the current model against the actual values for the validation dataset.
    Includes the base model's pre-computed predictions if base_model_eval_path is provided.
    """

    def __init__(
        self,
        validation_data_path: str,
        validation_batch_size: int = 32,
        plot_interval: int = 2,
        base_model_eval_path: str | None = None,
        plot_title_suffix: str = "",
    ):
        """Initialize the plotting callback.

        Args:
            validation_data_path: Path to the validation data file.
            validation_batch_size: Batch size for validation data.
            plot_interval: Interval (in epochs) for generating plots.
            base_model_eval_path: Path to the base model evaluation data file.
            plot_title_suffix: Suffix to add to plot title.
        """
        super().__init__()
        self.validation_data_path = validation_data_path
        self.validation_batch_size = validation_batch_size
        self.plot_interval = plot_interval
        self.base_model_eval_path = base_model_eval_path
        self.plot_title_suffix = plot_title_suffix

        # Load validation data (actuals and features)
        try:
            data = np.load(validation_data_path)
            self.x_val = data["x"]
            self.y_val = data["y"]
            logging.info(
                f"Loaded validation data for PlottingCallback from {validation_data_path}: "
                f"features shape {self.x_val.shape}, targets shape {self.y_val.shape}"
            )
        except FileNotFoundError:
            logging.error(
                f"Validation data not found at {validation_data_path} for PlottingCallback."
            )
            self.x_val, self.y_val = None, None
        except Exception as e:
            logging.error(
                f"Error loading validation data for PlottingCallback from {validation_data_path}: {e}"
            )
            self.x_val, self.y_val = None, None

        # Load pre-computed base model predictions if path is provided
        self.base_preds_for_this_data = None
        if self.base_model_eval_path and Path(self.base_model_eval_path).exists():
            try:
                eval_data = np.load(self.base_model_eval_path)
                # Assume the keys correspond to the data being plotted by *this* callback
                # This callback plots fine-tune validation data, so look for finetune_val_preds_base_model
                if "finetune_val_preds_base_model" in eval_data:
                    self.base_preds_for_this_data = eval_data[
                        "finetune_val_preds_base_model"
                    ]
                    logging.info(
                        f"Loaded base model predictions for fine-tune validation data from {self.base_model_eval_path}: shape {self.base_preds_for_this_data.shape}"
                    )
                    # Basic shape check
                    if (
                        self.x_val is not None
                        and self.base_preds_for_this_data.shape[0]
                        != self.x_val.shape[0]
                    ):
                        logging.warning(
                            f"Shape mismatch between loaded validation data ({self.x_val.shape[0]}) and base predictions ({self.base_preds_for_this_data.shape[0]}). Plotting base preds might fail."
                        )
                else:
                    logging.warning(
                        f"Key 'finetune_val_preds_base_model' not found in {self.base_model_eval_path}. Base curve will not be plotted for fine-tune data."
                    )
                    self.base_model_eval_path = (
                        None  # Disable base plotting if key not found
                    )

            except Exception as e:
                logging.error(
                    f"Error loading base model eval data from {self.base_model_eval_path}: {e}"
                )
                self.base_model_eval_path = None  # Disable base plotting on error
        elif self.base_model_eval_path:
            logging.warning(
                f"Base model evaluation file not found at {self.base_model_eval_path}. Base curve will not be plotted for fine-tune data."
            )
            self.base_model_eval_path = None  # Disable base plotting if file not found

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Plot predictions vs actuals at the end of a validation epoch."""
        # Check if plotting is scheduled for this epoch
        if (trainer.current_epoch + 1) % self.plot_interval != 0:
            return

        # Check if validation data was loaded successfully
        if self.x_val is None or self.y_val is None:
            logging.warning(
                "Validation data not available for PlottingCallback, skipping plotting."
            )
            return

        logging.info(
            f"Plotting predictions vs actuals for epoch {trainer.current_epoch + 1}"
            f"{self.plot_title_suffix}"  # Add suffix
        )

        # Get predictions from the current (fine-tuned) model
        pl_module.eval()  # Set module to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            current_preds = []
            # Ensure data is on the correct device for the current model
            device = pl_module.device
            for i in range(0, len(self.x_val), self.validation_batch_size):
                batch_x = torch.tensor(
                    self.x_val[i : i + self.validation_batch_size], dtype=torch.float32
                ).to(device)
                batch_preds = pl_module(batch_x).cpu().numpy()
                current_preds.append(batch_preds)
            current_preds = np.concatenate(current_preds, axis=0)

        pl_module.train()  # Set module back to training mode

        # Plotting
        try:
            # Use a clear figure name
            plt.figure(
                f"Validation Plot Epoch {trainer.current_epoch + 1}{self.plot_title_suffix}",
                figsize=(10, 6),
            )
            plt.scatter(
                self.x_val, self.y_val, label="Actual", alpha=0.5
            )  # Plot actual data

            # Plot fine-tuned model predictions
            plt.scatter(self.x_val, current_preds, label="Fine-tuned Preds", alpha=0.5)

            # Plot base model predictions if available for this dataset
            if self.base_preds_for_this_data is not None:
                plt.scatter(
                    self.x_val,
                    self.base_preds_for_this_data,
                    label="Base Preds",
                    alpha=0.5,
                )

            plt.title(
                f"Epoch {trainer.current_epoch + 1}: Predictions vs Actuals{self.plot_title_suffix}"
            )
            plt.xlabel("Input (x)")
            plt.ylabel("Output (y)")
            plt.legend()
            plt.grid(True)

            # Log plot to TensorBoard
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                fig = plt.gcf()
                fig.canvas.draw()
                plot_image = np.array(fig.canvas.renderer._renderer).transpose(2, 0, 1)[
                    :3
                ]  # Convert to C x H x W

                # Use a distinct tag for this plot
                trainer.logger.experiment.add_image(
                    f"Validation Predictions Plot{self.plot_title_suffix}",
                    plot_image,
                    trainer.global_step,
                )

            plt.close(fig)  # Close the specific figure to free memory

        except Exception as e:
            logging.error(
                f"Error during plotting callback{self.plot_title_suffix}: {e}"
            )


# --- New Callback for Base Data Comparison Plotting at the End of Training ---


class BaseDataComparisonPlottingCallback(Callback):
    """Callback for plotting predictions vs actuals on the BASE validation data at the end of training.

    Compares the fine-tuned model's predictions on base data with the base model's pre-computed predictions on base data.
    """

    def __init__(
        self,
        base_validation_data_path: str,
        base_model_eval_path: str,
        base_validation_batch_size: int = 32,
        plot_title_suffix: str = "",
    ):
        super().__init__()
        self.base_validation_data_path = base_validation_data_path
        self.base_validation_batch_size = base_validation_batch_size
        self.base_model_eval_path = base_model_eval_path
        self.plot_title_suffix = plot_title_suffix

        # Load base validation data (actuals and features) for comparison plot
        try:
            data = np.load(base_validation_data_path)
            self.x_base_val_plot = data["x"]
            self.y_base_val_plot = data["y"]
            logging.info(
                f"Loaded base validation data for BaseDataComparisonPlottingCallback from {base_validation_data_path}: "
                f"features shape {self.x_base_val_plot.shape}, targets shape {self.y_base_val_plot.shape}"
            )
        except FileNotFoundError:
            logging.error(
                f"Base validation data not found at {base_validation_data_path} for BaseDataComparisonPlottingCallback."
            )
            self.x_base_val_plot, self.y_base_val_plot = None, None
        except Exception as e:
            logging.error(
                f"Error loading base validation data from {base_validation_data_path}: {e}"
            )
            self.x_base_val_plot, self.y_base_val_plot = None, None

        # Load pre-computed base model predictions for base validation data
        self.base_preds_for_base_data = None
        if self.base_model_eval_path and Path(self.base_model_eval_path).exists():
            try:
                eval_data = np.load(self.base_model_eval_path)
                # This callback plots base validation data, so look for base_val_preds_base_model
                if "base_val_preds_base_model" in eval_data:
                    self.base_preds_for_base_data = eval_data[
                        "base_val_preds_base_model"
                    ]
                    logging.info(
                        f"Loaded base model predictions for base validation data from {self.base_model_eval_path}: shape {self.base_preds_for_base_data.shape}"
                    )
                    # Basic shape check
                    if (
                        self.x_base_val_plot is not None
                        and self.base_preds_for_base_data.shape[0]
                        != self.x_base_val_plot.shape[0]
                    ):
                        logging.warning(
                            f"Shape mismatch between loaded base validation data ({self.x_base_val_plot.shape[0]}) and base predictions ({self.base_preds_for_base_data.shape[0]}). Plotting base preds might fail."
                        )
                else:
                    logging.warning(
                        f"Key 'base_val_preds_base_model' not found in {self.base_model_eval_path}. Cannot plot base model comparison on base data."
                    )
                    self.base_model_eval_path = (
                        None  # Disable base plotting if key not found
                    )
            except Exception as e:
                logging.error(
                    f"Error loading base model eval data from {self.base_model_eval_path}: {e}"
                )
                self.base_model_eval_path = None  # Disable base plotting on error

        elif self.base_model_eval_path:
            logging.warning(
                f"Base model evaluation file not found at {self.base_model_eval_path}. Cannot plot base model comparison on base data."
            )
            self.base_model_eval_path = None  # Disable base plotting if file not found

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Plot comparison on base data at the end of training."""
        # Only run if base model eval data was loaded successfully
        if (
            self.x_base_val_plot is None
            or self.y_base_val_plot is None
            or self.base_preds_for_base_data is None
        ):
            logging.warning(
                "Base validation data or base model predictions not available for comparison plotting, skipping."
            )
            return

        logging.info(f"Plotting comparison on Base Validation Data after training.")

        # Get predictions from the current (fine-tuned) model on the BASE data
        pl_module.eval()  # Set module to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            current_preds_on_base_data = []
            # Ensure data is on the correct device for the current model
            device = pl_module.device
            for i in range(
                0, len(self.x_base_val_plot), self.base_validation_batch_size
            ):
                batch_x = torch.tensor(
                    self.x_base_val_plot[i : i + self.base_validation_batch_size],
                    dtype=torch.float32,
                ).to(device)
                batch_preds = pl_module(batch_x).cpu().numpy()
                current_preds_on_base_data.append(batch_preds)
            current_preds_on_base_data = np.concatenate(
                current_preds_on_base_data, axis=0
            )

        pl_module.train()  # Set module back to training mode

        # Plotting
        try:
            # Use a clear figure name
            plt.figure(
                f"Base Data Comparison Plot{self.plot_title_suffix}", figsize=(10, 6)
            )
            plt.scatter(
                self.x_base_val_plot,
                self.y_base_val_plot,
                label="Actual (Base Data)",
                alpha=0.5,
            )  # Plot actual base data

            # Plot fine-tuned model predictions on base data
            plt.scatter(
                self.x_base_val_plot,
                current_preds_on_base_data,
                label="Fine-tuned Preds (on Base Data)",
                alpha=0.5,
            )

            # Plot base model predictions on base data (pre-computed)
            plt.scatter(
                self.x_base_val_plot,
                self.base_preds_for_base_data,
                label="Base Preds (on Base Data)",
                alpha=0.5,
            )

            plt.title(
                f"Post-Training Comparison: Predictions vs Actuals{self.plot_title_suffix}"
            )
            plt.xlabel("Input (x)")
            plt.ylabel("Output (y)")
            plt.legend()
            plt.grid(True)

            # Log plot to TensorBoard
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                fig = plt.gcf()
                fig.canvas.draw()
                plot_image = np.array(fig.canvas.renderer._renderer).transpose(2, 0, 1)[
                    :3
                ]  # Convert to C x H x W

                # Use a distinct tag for this plot
                trainer.logger.experiment.add_image(
                    f"Base Data Comparison Plot{self.plot_title_suffix}",
                    plot_image,
                    trainer.global_step,  # Use global_step at end of training
                )

            plt.close(fig)  # Close the specific figure to free memory

        except Exception as e:
            logging.error(f"Error during BaseDataComparisonPlottingCallback: {e}")


# Assuming TestingCallback is defined here or elsewhere in this file
class TestingCallback(Callback):
    """Basic testing callback for illustration."""

    def __init__(
        self, test_data_path: str, test_batch_size: int = 32, test_interval: int = -1
    ):
        super().__init__()
        self.test_data_path = test_data_path
        self.test_batch_size = test_batch_size
        self.test_interval = test_interval  # -1 means no periodic testing

        # Load test data
        try:
            data = np.load(test_data_path)
            self.x_test = data["features"]
            self.y_test = data["targets"]
            logging.info(
                f"Loaded test data for TestingCallback from {test_data_path}: "
                f"features shape {self.x_test.shape}, targets shape {self.y_test.shape}"
            )
        except FileNotFoundError:
            logging.error(
                f"Test data not found at {test_data_path} for TestingCallback."
            )
            self.x_test, self.y_test = None, None
        except Exception as e:
            logging.error(
                f"Error loading test data for TestingCallback from {test_data_path}: {e}"
            )
            self.x_test, self.y_test = None, None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Run testing periodically based on interval."""
        if (
            self.test_interval > 0
            and (trainer.current_epoch + 1) % self.test_interval == 0
        ):
            self.run_test(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Run test once after trainer.test() is called."""
        # This is called automatically by trainer.test()
        pass  # You would typically log metrics here

    def run_test(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Perform test evaluation."""
        if self.x_test is None or self.y_test is None:
            logging.warning(
                "Test data not available for TestingCallback, skipping testing."
            )
            return

        logging.info(f"Running test evaluation at epoch {trainer.current_epoch + 1}")

        pl_module.eval()
        test_preds = []
        with torch.no_grad():
            device = pl_module.device
            for i in range(0, len(self.x_test), self.test_batch_size):
                batch_x = torch.tensor(
                    self.x_test[i : i + self.test_batch_size], dtype=torch.float32
                ).to(device)
                batch_preds = pl_module(batch_x).cpu().numpy()
                test_preds.append(batch_preds)
        test_preds = np.concatenate(test_preds, axis=0)

        # You would calculate metrics here (e.g., MSE) and log them
        # For simplicity, just log completion for now
        logging.info(f"Test evaluation completed for epoch {trainer.current_epoch + 1}")


# Add other implementation classes/functions here if they exist in your original file


def setup_nemo_logging():
    """Set up NeMo-compatible logging configuration."""
    # Configure logging to use NeMo's logging system
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set NeMo logging level
    nemo_logging.setLevel(nemo_logging.INFO)
