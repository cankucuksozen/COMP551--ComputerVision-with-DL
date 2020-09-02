import torch
import torchvision

from torch import Tensor
from torch import nn
from torch.nn import functional as F

class fully_connected(nn.Module):
    def __init__(self, input_dims, hidden_dims, out_dims, bias = True, drop = True):
        super(fully_connected, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.drop = drop
        
        self.fc1 = nn.Linear(input_dims, hidden_dims, bias = bias)
        self.activate = nn.LeakyReLU()
        if drop:
            self.drop = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(hidden_dims, out_dims, bias = bias)
        
        for i in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(i.weight, a=1)
            
    def forward(self, x):
        out = self.fc1(x)
        out = self.activate(out)
        if self.drop:
            out = self.drop(out)
        out = self.fc2(out)
        return out

