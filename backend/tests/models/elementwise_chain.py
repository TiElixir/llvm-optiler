"""Three elementwise ops in a row over a large tensor: fusable, but only if the pipeline fuses."""
import torch


class Chain(torch.nn.Module):
    def forward(self, x):
        a = x * 2.0
        b = a + 1.0
        return torch.relu(b)


model = Chain()
inputs = (torch.randn(1024, 1024),)
