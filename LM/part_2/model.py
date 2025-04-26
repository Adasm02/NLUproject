import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

# class for variational dropout
class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout
    
    def forward(self, x):
        # evaluation time, dropout is not applied
        if not self.training:
            return x
        
        batch_size = x.size(0)
        # create the mask
        mask = x.data.new(batch_size, 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask/(1-self.dropout)
        mask = mask.expand_as(x)
        # apply the mask
        return x * mask

class LSTMLanguageModel(nn.Module):
    """
    Implements an LSTM-based language model.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, weight_tying=False,
                 drop_out_mode ="standard", num_layers=1):
        
        super(LSTMLanguageModel, self).__init__()
        
        if drop_out_mode == "standard":
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
            self.embedding_dropout = nn.Dropout(emb_dropout)
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)
            self.output_dropout = nn.Dropout(out_dropout)
            self.pad_token = pad_index
            self.output_layer = nn.Linear(hidden_dim, vocab_size)
        else:
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)            
            self.embedding_dropout = VariationalDropout(emb_dropout)
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)                   
            self.output_dropout = VariationalDropout(out_dropout)
            self.pad_token = pad_index
            self.output_layer = nn.Linear(hidden_dim, vocab_size)


        if weight_tying and embedding_dim == hidden_dim:
            self.output_layer.weight = self.token_embedding.weight
            print("Weight tying done")
        else:
            print("Weight tying is not possible. Hidden size is not equal to embedding size.")
    
    def forward(self, input_tokens):
        """
        Processes the input sequence through the embedding, LSTM, and output layers.
        """
        embedded = self.token_embedding(input_tokens)
        dropped_embedding = self.embedding_dropout(embedded)
        lstm_output, _ = self.lstm_layer(dropped_embedding)
        dropped_output = self.output_dropout(lstm_output)
        final_output = self.output_layer(dropped_output).permute(0, 2, 1)
        return final_output