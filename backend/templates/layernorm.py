"""LayerNorm over the last dimension."""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        return self.norm(x)


model = LayerNorm()
inputs = (torch.randn(256, 512),)
