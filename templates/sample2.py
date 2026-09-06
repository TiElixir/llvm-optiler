import torch
import torch.nn as nn

class TestModel(nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        y = y + b
        y = y * 2.0
        y = y / 3.0
        y = torch.relu(y - 1.0)
        return y

model = TestModel()

inputs = (
    torch.randn(4, 4),
    torch.randn(4, 4),
    torch.randn(4),
)