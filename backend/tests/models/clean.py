"""One tiny elementwise op: nothing to fix, so no findings."""
import torch


class Clean(torch.nn.Module):
    def forward(self, x):
        return x + 1.0


model = Clean()
inputs = (torch.randn(64, 64),)
