# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Attention(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        # self.use_cuda = use_cuda
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.reset_parameters()
        self.use_cuda = True

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.attention_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, mask):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        # print(inputs.shape) # torch.Size([64, 199, 256])
        # print(mask.shape) # torch.Size([64, 199])
        logits = inputs.matmul(self.attention_vector) 
        unnorm_ai = (logits - logits.max()).exp() # torch.Size([64, 199])
        masked_weights = unnorm_ai * mask  # torch.Size([64, 199])
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence # torch.Size([64, 1])
        attentions = masked_weights.div(att_sums)   
        # apply attention weights
        representations = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs)) 
        
        return (representations, attentions if self.return_attention else None)
