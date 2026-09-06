import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return torch.relu(y + b)

# You must define 'model' and 'inputs'
model = MyModel()
inputs = (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4))
