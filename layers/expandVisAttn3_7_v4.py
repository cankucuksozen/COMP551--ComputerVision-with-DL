import torch
import torchvision

from torch import Tensor
from torch import nn
from .fully_connected import fully_connected
from .attn2d_v4 import attn2d
from .attn1d_v4 import attn1d
import math

class expandVisAttn3_7(nn.Module):
    
    """
    
    """
    
    def __init__(self, input_dims, hidden_dims, out_dims, Nh):
        super(expandVisAttn3_7, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.Nh = Nh
        self.head_dims = self.hidden_dims // self.Nh
        self.outer_kernel_size = 7
        self.inner_kernel_size = 3
        self.rel_h = nn.Embedding(self.outer_kernel_size, hidden_dims//2)
        self.rel_w = nn.Embedding(self.outer_kernel_size, hidden_dims//2)
        
        self.init_q_proj = fully_connected(input_dims, hidden_dims, 
                                           hidden_dims, bias = False)
        self.attn1x1 = attn2d(input_dims, 1, self.outer_kernel_size, Nh, hidden_dims, hidden_dims)
        
        self.attn3x3 = attn2d(input_dims, 3, self.outer_kernel_size, Nh, hidden_dims, hidden_dims)
        self.selfattn1 = attn1d(hidden_dims, 2, Nh, hidden_dims, hidden_dims)
        
        self.attn5x5 = attn2d(input_dims, 5, self.outer_kernel_size, Nh, hidden_dims, hidden_dims)
        self.selfattn2 = attn1d(hidden_dims, 3, Nh, hidden_dims, hidden_dims)
        
        self.attn7x7 = attn2d(input_dims, 7, self.outer_kernel_size, Nh, hidden_dims, hidden_dims)
        self.selfattn3 = attn1d(hidden_dims, 4, Nh, hidden_dims, hidden_dims)
        
        self.out_proj = fully_connected(hidden_dims, out_dims, out_dims, bias = False)

    def generate_mask(self, stage):
        mask = torch.zeros((self.outer_kernel_size,self.outer_kernel_size)).to(torch.long).to('cuda')
        if stage == 0:
            mask[3,3] = 1
        elif stage == 1:
            mask[2:5,2:5] = 1
            mask[3,3] = 0
        elif stage == 2:
            mask[1:6,1:6] = 1
            mask[2:5,2:5] = 0
        elif stage == 3:
            mask[0:7,0:7] = 1
            mask[1:6,1:6] = 0    
        ind = torch.nonzero(mask)
        ind_row = ind[:,0]
        ind_col = ind[:,1]
        return ind_row, ind_col
    
    def get_memory_block(self, x, center, stage):
        ind_row, ind_col = self.generate_mask(stage)
        memory = x[:,:,ind_row,ind_col] 
        memory = memory.permute(0,2,1)
        return memory
    
    def get_rel_hw(self, stage):
        indices = torch.arange(0, self.outer_kernel_size).to('cuda')
        rel_h = self.rel_h(indices)
        rel_h = rel_h.expand((self.outer_kernel_size, self.outer_kernel_size, -1))
        rel_w = torch.reshape(self.rel_w(indices), (1, self.outer_kernel_size, -1))
        rel_w = rel_w.expand((self.outer_kernel_size,self.outer_kernel_size, -1))
        ind_row, ind_col = self.generate_mask(stage)
        masked_rel_h = rel_h[ind_row,ind_col,:]
        masked_rel_w = rel_w[ind_row,ind_col,:]
        return masked_rel_h, masked_rel_w
        
    def forward(self, x):
        b, c, h, w = x.shape
        assert h == 7 and w == 7, "Input spatial dimensions must be of shape: 7"
        center_pos = 3
        
        mem1x1 = self.get_memory_block(x, center_pos, 0)
        rel_h1x1, rel_w1x1 = self.get_rel_hw(0)
        qi = self.init_q_proj(mem1x1)
        attn0, attn1_pred = self.attn1x1(qi, mem1x1, rel_h1x1, rel_w1x1)
        
        ##----------------------------------------------------------------------------
        
        mem3x3 = self.get_memory_block(x, center_pos, 1)
        rel_h3x3, rel_w3x3 = self.get_rel_hw(1)
        r1, c1 = self.generate_mask(1)
        attn1_pred = attn1_pred[:,r1,c1]
        attn1, attn2_pred = self.attn3x3(attn0, mem3x3, rel_h3x3, rel_w3x3, attn1_pred)
        
        sa_input1 = torch.cat((attn0,attn1), dim = 1)
        q2 = self.selfattn1(attn1, sa_input1, sa_input1)
        
        ##----------------------------------------------------------------------------
        
        mem5x5 = self.get_memory_block(x, center_pos, 2)
        rel_h5x5, rel_w5x5 = self.get_rel_hw(2)
        r2, c2 = self.generate_mask(2)
        attn2_pred = attn2_pred[:,r2,c2]
        attn2, attn3_pred = self.attn5x5(q2, mem5x5, rel_h5x5, rel_w5x5, attn2_pred)
        
        sa_input2 = torch.cat((sa_input1,attn2), dim = 1)
        q3 = self.selfattn2(attn2, sa_input2, sa_input2)
        
        ##----------------------------------------------------------------------------
        
        mem7x7 = self.get_memory_block(x, center_pos, 3)
        rel_h7x7, rel_w7x7 = self.get_rel_hw(3)
        r3, c3 = self.generate_mask(3)
        attn3_pred = attn3_pred[:,r3,c3]
        attn3, _ = self.attn7x7(q3, mem7x7, rel_h7x7, rel_w7x7, attn3_pred)
         
        sa_input3 = torch.cat((sa_input2, attn3), dim = 1)
        attn_out = self.selfattn3(attn3, sa_input3, sa_input3)
        
        attn_out = self.out_proj(attn_out)
        ##----------------------------------------------------------------------------
        
        return attn_out
