import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class CustomRNNCell(nn.Module):
    """
    Implements a basic RNN cell with linear transformations.
    """
    def __init__(self, hidden_dim, input_dim, output_dim, vocab_size, dropout_rate=0.1):
        super(CustomRNNCell, self).__init__()
        self.input_transform = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_transform = nn.Linear(hidden_dim, hidden_dim)
        self.output_transform = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.vocab_size = vocab_size
    
    def forward(self, prev_hidden, word_embedding):
        """
        Computes the next hidden state and output.
        """
        input_projection = self.input_transform(word_embedding)
        hidden_projection = self.hidden_transform(prev_hidden)
        hidden_state = self.activation(input_projection + hidden_projection)
        output = self.output_transform(hidden_state)
        return hidden_state, output

class RNNLanguageModel(nn.Module):
    """
    Implements a simple RNN-based language model.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, num_layers=1):
        super(RNNLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.rnn_layer = nn.RNN(embedding_dim, hidden_dim, num_layers, bidirectional=False)
        self.pad_token = pad_index
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_tokens):
        """
        Processes the input sequence through the embedding and RNN layers.
        """
        embedded = self.token_embedding(input_tokens)
        rnn_output, _ = self.rnn_layer(embedded)
        final_output = self.output_layer(rnn_output).permute(0, 2, 1)
        return final_output

class LSTMLanguageModel(nn.Module):
    """
    Implements an LSTM-based language model.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, num_layers=1):
        super(LSTMLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.output_dropout = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    
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