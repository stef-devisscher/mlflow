import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor):
        """Forward function for the model.
        
        :param x: The input tensor. Expected shape (B, 3, H, W).
        """
        return self.lin(x)