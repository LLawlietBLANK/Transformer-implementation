import math
import torch 
import torch.nn as nn


class InputEmbedding(nn.Module):

    def __init__(self, d_model : int , vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self , x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self , d_model : int, seq_len : int , dropout : float):
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        #Creating a vector of shape (seq_len , d_model)

        pe = torch.zeros(seq_len , d_model)

        #Creating a vector of shape (seq_len)
        position = torch.arange(0 , seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , dtype = torch.float) * (-math.log(10000.0) / d_model))

        #apply the sin function to even the indices in the d_model 
        pe[: , 0::2] = torch.sin(position * div_term)

        #apply the cos function to odd the indices in the d_model 
        pe[: , 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe' , pe)

    def forward(self , x):
        x = x + (self.pe[:, :x.shape(1) , :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self , esp : float = 10**6) -> None :
        super().__init__()
        self.esp = esp
        self.alpha = nn.Parameter(torch.ones(1)) #multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #added


    def forward(self , x):
        mean = x.mean(dim = -1 , keepdim = True)
        std = x.std(dim = -1 , keepdim = True)
        return self.alpha * (x - mean) / (std + self.esp) + self.bias

