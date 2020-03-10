import torch
import torch.nn as nn
import torch.nn.functional as F

class Copy(nn.Module):
    def __init__(self, context_dim, hidden_dim, output_dim):
        super().__init__()
        self.w = nn.Linear(context_dim + hidden_dim + output_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.w(x))
