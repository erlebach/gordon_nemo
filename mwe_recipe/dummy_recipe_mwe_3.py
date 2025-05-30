import os

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


# Define the training function at the top level
def top_level_train_fn(model, data_module, trainer_config):
    trainer = trainer_config.build()  # Build the trainer from the config
    trainer.fit(model, datamodule=data_module)


# 4. Define a function to create the Trainer configuration
def create_trainer_config(
    max_steps: int, dir: str, num_gpus_per_node: int
) -> run.Config:
    accelerator_type = "gpu" if torch.cuda.is_available() else "cpu"
    return run.Config(
        pl.Trainer,
        max_steps=max_steps,
        accelerator=accelerator_type,
        devices=num_gpus_per_node,
        default_root_dir=dir,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


# 5. Define the NeMo-Run recipe function
@run.cli.factory(name="dummy_recipe") # Using run.cli.factory as autoconvert might still be tricky
def dummy_recipe(
    max_steps: int = 10,
    dir: str = "./dummy_checkpoints",
    name: str = "dummy_experiment", # name argument is part of the recipe, not trainer
    num_nodes: int = 1, # num_nodes is typically for distributed setup, handled by executor
    num_gpus_per_node: int = 1,
) -> run.Partial:
    trainer_config = create_trainer_config(max_steps, dir, num_gpus_per_node)
    model_config = run.Config(DummyModel)
    data_module_config = run.Config(DummyDataModule)

    # Pass configurations to the top-level training function
    return run.Partial(
        top_level_train_fn,
        model=model_config,
        data_module=data_module_config,
        trainer_config=trainer_config,
    )


# 6. Define the executor (LocalExecutor with torchrun launcher)
def create_local_executor(devices: int = 1) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }
    return run.LocalExecutor(
        ntasks_per_node=devices,
        launcher="torchrun",
        env_vars=env_vars,
    )


# 7. Main function to run the recipe
def main():
    # Get the recipe configuration first
    # For CLI usage, NeMo-Run would parse CLI args and call the factory
    # Here, we call it directly as if it's being used programmatically
    # If using CLI `nemo_run dummy_recipe --max_steps=5`, then `dummy_recipe` is called by nemo_run
    
    # To run programmatically as intended by `run.run()`:
    # We need a `Partial` that `run.run()` can execute.
    # The `dummy_recipe` factory should return this.
    
    recipe_partial = dummy_recipe() # This now returns a Partial of top_level_train_fn

    executor = create_local_executor(devices=1)
    run.run(recipe_partial, executor=executor, name="dummy_experiment_run")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
