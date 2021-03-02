import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    '''
    a basic positional encoding for transformers, taken directly from the pytorch tutorial
    on the subject, updated to use BATCH / LENGTH / FEATURE ordering instead of LENGTH /
    BATCH / FEATURE which is the pytorch default
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == '__main__':

    x = PositionalEncoding(4, max_len=500)

    inp = torch.rand(50, 128, 4)

    x(inp)
