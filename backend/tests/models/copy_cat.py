"""Concatenate x with a computed tensor: the computed half is built in a temporary and then copied."""
import torch


class CopyCat(torch.nn.Module):
    def forward(self, x):
        y = torch.cat([x, x * 2.0], dim=0)
        return torch.relu(y)


model = CopyCat()
inputs = (torch.randn(512, 512),)
