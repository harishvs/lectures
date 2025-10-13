import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule

# Initialize distributed environment for single process
if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="nccl", rank=0, world_size=1)

# Define a simple Transformer model
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(1000, 512)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=512, nhead=8)
            for _ in range(3)
        ])
        self.output = nn.Linear(512, 1000)
    
    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

model = Transformer()
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

assert isinstance(model, Transformer)
assert isinstance(model, FSDPModule)
print(model)
#  FSDPTransformer(
#    (tok_embeddings): Embedding(...)
#    ...
#    (layers): 3 x FSDPTransformerBlock(...)
#    (output): Linear(...)
#  )

# Clean up distributed process group
dist.destroy_process_group()