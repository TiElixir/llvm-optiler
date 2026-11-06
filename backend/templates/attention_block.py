"""Single-head self-attention block (x: 128x64)."""
import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = q @ k.transpose(0, 1)
        scores = scores * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        return out


model = SelfAttention()
inputs = (torch.randn(128, 64),)
