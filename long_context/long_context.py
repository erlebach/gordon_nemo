import torch
import torch.nn as nn
import pytorch_lightning as pl
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

class FeedForwardBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class LongContextFFN(pl.LightningModule):
    def __init__(self, dim, num_blocks, lr=1e-3):
        super().__init__()
        self.blocks = nn.Sequential(*[FeedForwardBlock(dim) for _ in range(num_blocks)])
        self.output = nn.Linear(dim, 1)
        self.lr = lr

    def forward(self, x):
        return self.output(self.blocks(x)).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Synthetic DataLoader
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, dim, num_samples=100):
        self.x = torch.randn(num_samples, seq_len, dim)
        self.y = torch.randn(num_samples, seq_len)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Parameters
SEQ_LEN = 32768  # Very long context
DIM = 512
NUM_BLOCKS = 8
BATCH_SIZE = 2
NUM_GPUS = 2  # Set to your available GPU count

# DataLoader
dataset = SyntheticDataset(SEQ_LEN, DIM)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

# NeMo context parallelism strategy
strategy = NLPDDPStrategy(
    context_parallel_size=NUM_GPUS,  # Enable context parallelism
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

# Trainer
trainer = pl.Trainer(
    devices=NUM_GPUS,
    accelerator="gpu",
    strategy=strategy,
    max_epochs=1,
)

# Model
model = LongContextFFN(DIM, NUM_BLOCKS)

# Train
trainer.fit(model, loader)
