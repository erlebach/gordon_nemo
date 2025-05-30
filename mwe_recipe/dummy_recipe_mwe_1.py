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


# 4. Define the NeMo-Run recipe function returning a run.Partial
@run.autoconvert(partial=True)
def dummy_recipe(
    max_steps: int = 10,
    dir: str = "./dummy_checkpoints",
    name: str = "dummy_experiment",
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
) -> run.Partial:
    # Configure the PyTorch Lightning trainer
    accelerator_type = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=accelerator_type,
        devices=num_gpus_per_node,
        default_root_dir=dir,
        logger=False,  # Disable logging for simplicity
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # Return a run.Partial wrapping the training function
    def train_fn():
        model = DummyModel()
        data = DummyDataModule()
        trainer.fit(model, datamodule=data)

    return run.Partial(train_fn)


# 5. Define the executor (LocalExecutor with torchrun launcher)
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


# 6. Main function to run the recipe
def main():
    recipe = dummy_recipe()
    executor = create_local_executor(devices=1)
    run.run(recipe, executor=executor, name="dummy_experiment_run")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
