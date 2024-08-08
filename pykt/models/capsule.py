# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .attentionlayer import Attention


class Capsule(nn.Module):
    def __init__(self, dim_vector, final_dropout_rate,  num_c):
        super(Capsule, self).__init__()
        self.dim_vector = dim_vector
        self.add_module('attention_layer', Attention(attention_size=dim_vector, return_attention=True))
        self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
      
        self.add_module('linear_prob', nn.Linear(dim_vector, num_c))
        
        self.add_module('linear_r_si', nn.Linear(num_c, dim_vector))
       

    def forward(self, matrix_hidden_pad, mask=None):
        v_ci, attention = self.attention_layer(matrix_hidden_pad, mask)  
      
        prob = F.sigmoid(self.linear_prob(self.final_dropout(v_ci)))  
       
        r_si = self.linear_r_si(self.final_dropout(prob)) * v_ci
        
        return prob, r_si, attention 
       

