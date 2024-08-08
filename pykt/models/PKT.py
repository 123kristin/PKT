import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout


from .EncoderRNN import EncoderRNN
import torch.nn.functional as F
from torch import optim

class rnnCapsule(Module):
    # def __init__(self, num_c, emb_size, batch_size, n_layer, n_label, embed_dropout, cell_dropout, final_dropout, bidirectional, \
    #              rnn_type, emb_type='qid',  emb_path="", pretrain_dim=768):
   
    def __init__(self, num_c, num_q, emb_size, batch_size, n_layer, n_label, embed_dropout, cell_dropout, out_dropout, final_dropout, bidirectional, \
                 rnn_type, emb_type='qid',  emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "PKT"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
 
        self.n_layer = n_layer
        self.n_label = n_label
        self.embed_dropout = embed_dropout
        self.cell_dropout = cell_dropout
        self.out_dropout = out_dropout
        
       
        self.final_dropout = final_dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.batch_size = batch_size
   
        self.num_q = num_q


      


        # self.model = EncoderRNN(emb_size, self.hidden_size, n_layer, n_label, num_c, emb_type,
        #         embed_dropout, cell_dropout, final_dropout, bidirectional, rnn_type)
    
        self.model = EncoderRNN(emb_size, self.hidden_size, n_layer, n_label, num_c, num_q, emb_type,
                embed_dropout, cell_dropout, out_dropout, final_dropout, bidirectional, rnn_type)

  

  

    def stepTrain(self, c, q, r, m, device):
        self.model = self.model.to(device)
       
       
        input_variable = c
      
        input_q = q
   
        tensor_label = r 
  
        mask = m
        batch_size = c.shape[0]
    
        hidden = self.model.init_hidden(batch_size, device)
        
       

        # loss_sim, prob = self.model(input_variable, hidden, mask, tensor_label, device)
       
        loss_sim, prob = self.model(input_variable, input_q, hidden, mask, tensor_label, device)
       
  


      
        return loss_sim, prob 
