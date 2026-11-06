"""Row-wise softmax over a 2-D tensor."""
import torch
import torch.nn as nn


class Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


model = Softmax()
inputs = (torch.randn(256, 512),)
