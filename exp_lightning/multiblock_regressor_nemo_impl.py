"""NeMo-compatible implementation for MultiBlock Regressor callbacks and utilities.

This module provides NeMo-compatible callbacks and utilities for the MultiBlock Regressor,
adapted from the Lightning implementation to work with NeMo's framework.
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from nemo.utils import logging as nemo_logging


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
    config: dict[str, Any], test_data_path: str | None = None
) -> list[Callback]:
    """Create NeMo-compatible callbacks based on configuration.

    Args:
        config: Configuration dictionary.
        test_data_path: Optional path to test data for plotting.

    Returns:
        List of configured callbacks (excluding ModelCheckpoint which is handled by exp_manager).
    """
    callbacks = []

    # Always add loss history callback
    loss_history = NeMoLossHistory()
    callbacks.append(loss_history)

    # Add prediction plotter if test data is available
    if test_data_path and os.path.exists(test_data_path):
        plot_interval = config.get("trainer", {}).get("plot_interval", 2)
        prediction_plotter = NeMoPredictionPlotter(
            test_data_path=test_data_path,
            loss_history_callback=loss_history,
            plot_interval=plot_interval,
        )
        callbacks.append(prediction_plotter)
        nemo_logging.info(f"Added prediction plotter with test data: {test_data_path}")
    else:
        nemo_logging.info(
            "No test data path provided or file not found. Skipping prediction plotter."
        )

    # NOTE: Do NOT add ModelCheckpoint here - exp_manager handles that
    return callbacks


def setup_nemo_logging():
    """Set up NeMo-compatible logging configuration."""
    # Configure logging to use NeMo's logging system
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set NeMo logging level
    nemo_logging.setLevel(nemo_logging.INFO)
