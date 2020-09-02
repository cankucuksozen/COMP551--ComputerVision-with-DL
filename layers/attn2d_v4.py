import torch
import torchvision

from torch import Tensor
from torch import nn
from torch.nn import functional as F

import math

from .fully_connected import fully_connected

class attn2d(nn.Module):
    
    """
    
    """
    
    def __init__(self, input_dims, kernel_size, outer_kernel_size, Nh, dk, dv):
        super(attn2d, self).__init__()
        
        self.input_dim = input_dims
        self.kernel_size = kernel_size
        self.outer_kernel_size = outer_kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh
        
        self.W_k = nn.Linear(input_dims, dk, bias = False)
        self.W_v = nn.Linear(input_dims, dv, bias = False)
        self.softmax = nn.Softmax(dim=3)
        
        self.W_z = nn.Linear(dv, dv, bias = False)
        self.layernorm1 = nn.LayerNorm(dv)
        self.fc = fully_connected(dv, dv, dv, bias =True)
        self.layernorm2 = nn.LayerNorm(dv)
        
        if self.outer_kernel_size == 7:
            self.predict_next_fc = fully_connected(dv, dv//2, (self.kernel_size + 2)**2, bias = False) 
        elif self.outer_kernel_size == 15:
            self.predict_next_fc = fully_connected(dv, dv//2, (self.kernel_size + 4)**2, bias = False) 
        
        for i in [self.W_k, self.W_v, self.W_z]:
            nn.init.kaiming_normal_(i.weight, a=1)
            
    def forward(self, q, memory, rel_h, rel_w, prev_attn = None): ###-------------
        
        residual = q
        
        b, hw, c = memory.shape
        
        k = self.W_k(memory)
        v = self.W_v(memory)

        q = self.split_heads_2d(q, self.Nh)
        k = self.split_heads_2d(k, self.Nh)
        v = self.split_heads_2d(v, self.Nh)
                
        q *= self.dkh ** -0.5    
        k = k.permute(0,1,3,2)    
        logits = torch.matmul(q, k)
    
        rel_logits = self.rel_logits_2d(q, rel_h, rel_w)
        
        logits += rel_logits
        if prev_attn is not None:
            n = prev_attn.shape[-1]
            prev_attn = prev_attn.expand((self.Nh, 1, -1, -1)).permute(2,0,1,3)
            logits += prev_attn
        weights = self.softmax(logits)
        
        attn = torch.matmul(weights, v)
        attn = self.combine_heads_2d(attn)
        
        attn = self.W_z(attn)
        
        attn_pred = self.predict_next_attn(attn)
        
        out1 = attn + residual 
        out1 = self.layernorm1(out1)
        out2 = self.fc(out1)
        out2 = out2 + out1
        out2 = self.layernorm2(out2)
        
        return out2, attn_pred
        
    def split_heads_2d(self, x, Nh):
        b, n, d = x.shape
        ret_shape = (b, self.Nh, n, d // self.Nh)
        out = torch.reshape(x, ret_shape)
        return out

    def rel_logits_2d(self, q, rel_h, rel_w):
        b, Nh, _, dh = q.shape
        rel_h = torch.reshape(rel_h, (-1, Nh, dh//2)).permute(1,0,2)
        rel_w = torch.reshape(rel_w, (-1, Nh, dh//2)).permute(1,0,2)
        rel = torch.cat((rel_h,rel_w),dim=2)
        rel = rel.expand((b,-1,-1,-1))
        rel_logits = rel * q
        rel_logits = torch.reshape(torch.sum(rel_logits, dim=-1),(b,Nh,1,-1))
        return rel_logits

    def combine_heads_2d(self, x):
        b, Nh, hw, d = x.shape
        ret_shape = (b, hw, Nh*d)
        out = torch.reshape(x, ret_shape)
        return out
    
    def predict_next_attn(self, attn):
        b, _, d = attn.shape
        
        if self.outer_kernel_size == 7:
            blank = torch.zeros((b,7,7)).to('cuda')
            attn_pred = self.predict_next_fc(attn)
            attn_pred = torch.reshape(attn_pred, (b, self.kernel_size + 2, self.kernel_size + 2))
            if self.kernel_size == 1:
                attn_pred = F.pad(attn_pred, (2,2,2,2), "constant", 0)
            elif self.kernel_size == 3:
                attn_pred = F.pad(attn_pred, (1,1,1,1), "constant", 0)
            elif self.kernel_size == 5:
                attn_pred = blank
            return attn_pred
        
        elif self.outer_kernel_size == 15:
            blank = torch.zeros((b,15,15)).to('cuda')
            attn_pred = self.predict_next_fc(attn)
            attn_pred = torch.reshape(attn_pred, (b, self.kernel_size + 4, self.kernel_size + 4))
            if self.kernel_size == 3:
                attn_pred = F.pad(attn_pred, (6,6,6,6), "constant", 0)
            elif self.kernel_size == 7:
                attn_pred = F.pad(attn_pred, (4,4,4,4), "constant", 0)
            elif self.kernel_size == 11:
                attn_pred = F.pad(attn_pred, (2,2,2,2), "constant", 0)
            elif self.kernel_size == 15:
                attn_pred = blank
            return attn_pred
        
    
    """
    def predict_next_attn(self, attn):
        b, _, d = attn.shape
        blank = torch.zeros((b,15,15)).to('cuda')
        attn_pred = self.predict_next_fc(attn)
        attn_pred = torch.reshape(attn_pred, (b, self.kernel_size + 4, self.kernel_size + 4))
        if self.kernel_size == 3:
            attn_pred = F.pad(attn_pred, (6,6,6,6), "constant", 0)
        elif self.kernel_size == 7:
            attn_pred = F.pad(attn_pred, (4,4,4,4), "constant", 0)
        elif self.kernel_size == 11:
            attn_pred = F.pad(attn_pred, (2,2,2,2), "constant", 0)
        elif self.kernel_size == 15:
            blank = attn_pred
     
        
    """   
