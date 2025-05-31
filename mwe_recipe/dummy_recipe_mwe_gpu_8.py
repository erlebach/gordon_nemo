import os
import time
from dataclasses import dataclass

import nemo_run as run
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# 1. Define a trivial dataset yielding fixed dummy data
class DummyDataset(Dataset):
    def __len__(self):
        return 100  # arbitrary small size

    def __getitem__(self, idx):
        # Return a fixed input tensor and target tensor
        x = torch.zeros(10)  # input vector of zeros
        y = torch.zeros(1)  # target scalar zero
        return x, y


# 2. Define a minimal DataModule using the dummy dataset
class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=4)


# 3. Define a trivial LightningModule that predicts zero regardless of input
class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # A single linear layer initialized to zero weights and bias
        self.linear = torch.nn.Linear(10, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


# 4. Define the training function at the top level
def train_function(trainer, model, data_module):
    trainer.fit(model, datamodule=data_module)


# 5. Define a dataclass for the recipe configuration
@dataclass
class DummyRecipeConfig:
    max_steps: int = 10
    dir: str = "./dummy_checkpoints"
    name: str = "dummy_experiment"
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    accelerator: str = "cpu"  # Hardcoded to cpu for simplicity

    def __post_init__(self):
        # Optionally, resolve accelerator type during initialization if needed
        if self.accelerator == "auto":
            self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"


# 6. Define the NeMo-Run recipe function using run.autoconvert
@run.autoconvert(partial=True)
def dummy_recipe(
    max_steps: int = 10,
    dir: str = "./dummy_checkpoints",
    name: str = "dummy_experiment",
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    # @run.autoconvert does not allow
    accelerator: str = "gpu",  # Hardcoded to gpu for now
) -> run.Partial:
    # Create recipe configuration using dataclass
    config = DummyRecipeConfig(
        max_steps=max_steps,
        dir=dir,
        name=name,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        accelerator=accelerator,
    )

    # Configure the trainer as a run.Config object using dataclass values
    trainer_config = run.Config(
        pl.Trainer,
        max_steps=config.max_steps,
        accelerator=config.accelerator,
        devices=1,  # Explicitly set to 1 to avoid multi-device setup
        default_root_dir=config.dir,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        strategy="auto",  # Let PyTorch Lightning handle strategy, avoid DDP if not needed
    )

    # Configure model and data module as run.Config objects
    model_config = run.Config(DummyModel)
    data_module_config = run.Config(DummyDataModule)

    # Return a run.Partial of the training function
    return run.Partial(
        train_function,
        trainer=trainer_config,
        model=model_config,
        data_module=data_module_config,
    )


# 7. Define the executor (LocalExecutor with no launcher)
def create_local_executor(devices: int = 1) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }
    return run.LocalExecutor(
        ntasks_per_node=1,  # Explicitly set to 1 to avoid multi-process setup
        # Avoid torchrun if not connected to the internet.
        # launcher="torchrun",
        launcher=None,  # Set to None to avoid torchrun and distributed setup
        env_vars=env_vars,
    )


# 8. Main function to run the recipe
def main():
    recipe = dummy_recipe()
    # Add a timestamp to the experiment name to ensure uniqueness
    experiment_name = f"dummy_experiment_run_{int(time.time())}"
    executor = create_local_executor(devices=1)
    run.run(recipe, executor=executor, name=experiment_name)
    return 0


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
    print("exit code")
