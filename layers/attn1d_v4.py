import torch
import torchvision

from torch import Tensor
from torch import nn
from torch.nn import functional as F

import math
from .fully_connected import fully_connected

class attn1d(nn.Module):
    
    """
    
    """
    
    def __init__(self, input_dims, seq_len, Nh, dk, dv):
        super(attn1d, self).__init__()
        
        self.input_dim = input_dims
        self.seq_len = seq_len
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh
        
        self.W_q = nn.Linear(input_dims, dk, bias = False)
        self.W_k = nn.Linear(input_dims, dk, bias = False)
        self.W_v = nn.Linear(input_dims, dv, bias = False)
        
        self.W_z = nn.Linear(dv, dv, bias = False)
    
        self.softmax = nn.Softmax(dim=-1) 
        
        self.layernorm1 = nn.LayerNorm(dv)
        self.fc = fully_connected(dv, dv, dv, bias = True)
        self.layernorm2 = nn.LayerNorm(dv)
                               
        for i in [self.W_q, self.W_k, self.W_v, self.W_z]:
            nn.init.kaiming_normal_(i.weight, a=1)
        
        self.rel_pos =  nn.Embedding(seq_len, dk)         
            
    def forward(self, qi, ki, vi): ###-------------
        residual = qi
        
        b, n, c = ki.shape
        
        q = self.W_q(qi)
        k = self.W_k(ki)
        v = self.W_v(vi)

        q = self.split_heads_1d(q, self.Nh)
        k = self.split_heads_1d(k, self.Nh)
        v = self.split_heads_1d(v, self.Nh)
                
        q *= self.dkh ** -0.5    
        k = k.permute(0,1,3,2)   
        logits = torch.matmul(q, k)
        
        rel_logits = self.rel_logits_1d(q, v)
        logits += rel_logits
        
        weights = self.softmax(logits)
        attn = torch.matmul(weights, v)
        attn = self.combine_heads_1d(attn)
        
        attn = self.W_z(attn)
        
        out1 = attn + residual 
        out1 = self.layernorm1(out1)
        out2 = self.fc(out1)
        out2 = out2 + out1
        out2 = self.layernorm2(out2)
        
        return out2
        
    def split_heads_1d(self, x, Nh):
        b, n, d = x.shape
        ret_shape = (b, self.Nh, n, d // self.Nh)
        out = torch.reshape(x, ret_shape)
        return out

    def rel_logits_1d(self, q, v):
        b, Nh, n, dh = v.shape
        indices = torch.arange(0,n).to(v.device)
        rel = self.rel_pos(indices)
        rel = torch.reshape(rel, (-1, self.Nh, self.dkh)).permute(1,0,2)
        rel = rel.expand((b,-1,-1,-1))
        rel_logits = torch.matmul(q, rel.permute(0,1,3,2))
        return rel_logits

    def combine_heads_1d(self, x):
        b, Nh, hw, d = x.shape
        ret_shape = (b, hw, Nh*d)
        out = torch.reshape(x, ret_shape)
        return out
