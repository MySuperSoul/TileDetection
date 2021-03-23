import torch
import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, scale_ratio=1.0):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.scale = nn.Parameter(
            torch.tensor([self.scale_ratio], dtype=torch.float32),
            requires_grad=True)

    def forward(self, feat):
        return feat * self.scale