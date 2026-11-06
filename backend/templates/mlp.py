"""Three-layer MLP with ReLU."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        return self.fc3(h)


model = MLP()
inputs = (torch.randn(128, 64),)
