"""Matmul, bias add and ReLU (256x256)."""
import torch
import torch.nn as nn


class MatmulRelu(nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return torch.relu(y + b)


model = MatmulRelu()
inputs = (torch.randn(256, 256), torch.randn(256, 256), torch.randn(256))
