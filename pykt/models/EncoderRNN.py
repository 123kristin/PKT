# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from .capsule import Capsule
from torch.nn import Embedding, Linear
import pickle
import os

class EncoderRNN(nn.Module):
  
    def __init__(self, dim_input, dim_hidden, n_layers, n_label, num_c, num_q, emb_type, embed_dropout_rate, \
                 cell_dropout_rate, out_dropout_rate, final_dropout_rate, bidirectional, rnn_type):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.n_label = n_label
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.emb_type = emb_type
        self.use_cuda = True
        

        self.num_c = num_c
        
  
        self.num_q = num_q

  
 

        if emb_type.startswith("qid"):
         
            self.interaction_emb = Embedding(self.num_c * 2, dim_input)
            
            

        self.add_module('embed_dropout', nn.Dropout(embed_dropout_rate))
        self.add_module('rnn', getattr(nn, self.rnn_type)(dim_input, dim_hidden, n_layers, batch_first=True, 
                dropout=cell_dropout_rate, bidirectional=bidirectional,))
        
        self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
        self.add_module('out_dropout', nn.Dropout(out_dropout_rate))
      
        self.add_module('out_layer', nn.Linear(dim_input, dim_input))
        self.add_module('linear_sim', nn.Linear(dim_input, num_c))

        for i in range(self.n_label):
            self.add_module('capsule_%s' % i, Capsule(dim_hidden * (2 if self.bidirectional else 1), final_dropout_rate, self.num_c))

      
    def forward(self, input, input_q, hidden, mask, tensor_label, device):
       
        c = input
        
        r = tensor_label
        emb_type = self.emb_type # 'qid'
        if emb_type == "qid":
            x = c + self.num_c * r 
           
            embedded = self.interaction_emb(x)
           
        embedded = self.embed_dropout(embedded) 
       

        output, hidden = self.rnn(embedded, hidden)
        

   
    
        actual_lengths = mask.sum(dim=1, keepdim=True)  
   
        variable_len = 1.0 / actual_lengths.float()
        variable_len = variable_len.to(device)
        


        v_s = torch.sum(output, 1) * variable_len 
        
      
        
        

        list_prob, list_r_s, list_attention = [], [], []
        print(mask.shape) 
        for i in range(self.n_label):
            prob_tmp, r_s_tmp, attention = getattr(self, 'capsule_%s' % i)(output, mask) 
            print(attention.shape)
            list_prob.append(prob_tmp) 
            list_r_s.append(r_s_tmp) 
            list_attention.append(attention) 
            print(list_attention[0].shape)
            

        list_r_s = torch.stack(list_r_s)  
        list_prob = torch.stack(list_prob)
        prob = torch.mean(list_prob, dim=0) 

 

        list_r_s, _ = torch.max(list_r_s, dim=0)
        output = self.out_layer(self.out_dropout(output))
        
        list_sim = F.sigmoid(self.linear_sim(self.final_dropout(output * list_r_s)))


        return list_sim, prob

        

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        h_0 = Variable(weight.new(self.n_layers * (2 if self.bidirectional else 1), batch_size, self.dim_hidden).zero_(), requires_grad=False)
        h_0 = h_0.to(device)
        return (h_0, h_0) if self.rnn_type == "LSTM" else h_0
