"""Generate data for base and finetuned models"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np


class DataBase:
    @classmethod
    def create_sine_data(cls, n=1000, noise=0.01):
        """Create sine wave dataset.

        Args:
            n: Number of data points
            noise: Noise level

        Returns:
            Tuple of inputs and targets
        """
        x = np.linspace(-np.pi, np.pi, n).reshape(-1, 1)
        y = np.sin(x) + noise * np.random.randn(*x.shape)
        return x, y

    @classmethod
    def prepare_data(cls, train_size=1000, val_size=200, test_size=200, noise=0.01):
        """Prepare datasets for training, validation and testing.

        Args:
            train_size: Size of training set
            val_size: Size of validation set
            test_size: Size of test set
            noise: Noise level
        """
        # Create directories if they don't exist
        os.makedirs("data/base", exist_ok=True)

        # Generate data
        x_train, y_train = cls.create_sine_data(n=train_size, noise=noise)
        x_val, y_val = cls.create_sine_data(n=val_size, noise=noise)
        x_test, y_test = cls.create_sine_data(n=test_size, noise=noise)

        # Save datasets
        np.savez("data/base/sine_train.npz", x=x_train, y=y_train)
        np.savez("data/base/sine_val.npz", x=x_val, y=y_val)
        np.savez("data/base/sine_test.npz", x=x_test, y=y_test)

        logging.info("Datasets created and saved to the 'data' directory.")


class DataFinetune:
    @classmethod
    def create_modified_sine_data(cls, n=1000, noise=0.1, phase=0.5):
        """Create modified sine wave dataset with phase shift.

        Args:
            n: Number of data points
            noise: Noise level
            phase: Phase shift

        Returns:
            Tuple of inputs and targets
        """
        x = np.linspace(-np.pi, np.pi, n).reshape(-1, 1)
        y = np.sin(x + phase) + noise * np.random.randn(*x.shape)
        return x, y

    @classmethod
    def prepare_finetune_data(
        cls, train_size=1000, val_size=200, test_size=200, noise=0.01, phase=0.5
    ):
        """Prepare datasets for fine-tuning.

        Args:
            train_size: Size of training set
            val_size: Size of validation set
            test_size: Size of test set
            noise: Noise level
            phase: Phase shift
        """
        # Create directories if they don't exist
        os.makedirs("data/finetune", exist_ok=True)

        # Generate data
        # Same sine wave for all three, different noise
        x_train, y_train = cls.create_modified_sine_data(
            n=train_size, noise=noise, phase=phase
        )
        x_val, y_val = cls.create_modified_sine_data(
            n=val_size, noise=noise, phase=phase
        )
        x_test, y_test = cls.create_modified_sine_data(
            n=test_size, noise=noise, phase=phase
        )

        # Save datasets
        np.savez("data/finetune/sine_train.npz", x=x_train, y=y_train)
        np.savez("data/finetune/sine_val.npz", x=x_val, y=y_val)
        np.savez("data/finetune/sine_test.npz", x=x_test, y=y_test)

        logging.info(
            "Fine-tuning datasets created and saved to the 'data/finetune' directory."
        )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate data for both base and finetune
    print("Generating Base Data...")
    DataBase.prepare_data()
    print("\nGenerating Finetune Data...")
    # Use the phase shift from the finetune config, assuming a default if not specified
    phase_shift_for_test = 0.5
    DataFinetune.prepare_finetune_data(phase=phase_shift_for_test)

    # --- Visualize Data ---
    print("\nVisualizing Data...")

    # Load data for plotting
    base_train_data = np.load("data/base/sine_train.npz")
    base_val_data = np.load("data/base/sine_val.npz")
    base_test_data = np.load("data/base/sine_test.npz")

    finetune_train_data = np.load("data/finetune/sine_train.npz")
    finetune_val_data = np.load("data/finetune/sine_val.npz")
    finetune_test_data = np.load("data/finetune/sine_test.npz")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Adjusted figsize for subplots

    # --- Plot Base Data ---
    axes[0].plot(
        base_train_data["x"],
        base_train_data["y"],
        "bo",
        markersize=3,
        alpha=0.5,
        label="Train Data",  # Changed label for clarity in subplot
    )
    axes[0].plot(
        base_val_data["x"],
        base_val_data["y"],
        "go",
        markersize=3,
        alpha=0.5,
        label="Validation Data",  # Changed label for clarity in subplot
    )
    axes[0].plot(
        base_test_data["x"],
        base_test_data["y"],
        "ro",
        markersize=3,
        alpha=0.5,
        label="Test Data",  # Changed label for clarity in subplot
    )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Base Sine Wave Datasets")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot Finetune Data ---
    axes[1].plot(
        finetune_train_data["x"],
        finetune_train_data["y"],
        "bo",
        markersize=3,
        alpha=0.5,
        label="Train Data",  # Changed label for clarity in subplot
    )
    axes[1].plot(
        finetune_val_data["x"],
        finetune_val_data["y"],
        "go",
        markersize=3,
        alpha=0.5,
        label="Validation Data",  # Changed label for clarity in subplot
    )
    axes[1].plot(
        finetune_test_data["x"],
        finetune_test_data["y"],
        "ro",
        markersize=3,
        alpha=0.5,
        label="Test Data",  # Changed label for clarity in subplot
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"Finetune (Phase={phase_shift_for_test}) Sine Wave Datasets")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Set Same Axis Limits ---
    # Determine overall min/max for x and y across both datasets
    all_x = np.concatenate(
        [
            base_train_data["x"],
            base_val_data["x"],
            base_test_data["x"],
            finetune_train_data["x"],
            finetune_val_data["x"],
            finetune_test_data["x"],
        ]
    )
    all_y = np.concatenate(
        [
            base_train_data["y"],
            base_val_data["y"],
            base_test_data["y"],
            finetune_train_data["y"],
            finetune_val_data["y"],
            finetune_test_data["y"],
        ]
    )

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # Add a small buffer to the limits
    x_buffer = (x_max - x_min) * 0.05
    y_buffer = (y_max - y_min) * 0.1

    axes[0].set_xlim(x_min - x_buffer, x_max + x_buffer)
    axes[0].set_ylim(y_min - y_buffer, y_max + y_buffer)
    axes[1].set_xlim(x_min - x_buffer, x_max + x_buffer)
    axes[1].set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Save the combined figure
    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    plt.savefig("combined_sine_data_visualization.png")  # Changed output filename
    plt.close()
    print(
        "Combined data visualization saved to 'combined_sine_data_visualization.png'."
    )
