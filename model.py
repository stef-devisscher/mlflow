import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor):
        """Forward function for the model.
        
        :param x: The input tensor.
        """
        return self.lin(x)

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.lin1 = nn.Linear(2, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """Forward function for the model.
        
        :param x: The input tensor.
        """
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.lin1 = nn.Linear(2, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.lin4 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """Forward function for the model.
        
        :param x: The input tensor.
        """
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        return x