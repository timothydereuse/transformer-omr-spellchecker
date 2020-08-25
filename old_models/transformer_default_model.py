
import numpy as np
import torch
import torch.nn as nn
import load_lmd as lmd

feats = 256     # num. features dimension
nhid = 100      # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2     # number of decoder/encoder layers
nhead = 2       # the number of heads in the multiheadattention models
dropout = 0.1   # the dropout value

model = nn.Transformer(
    d_model=feats,
    nhead=nhead,
    num_encoder_layers=nlayers,
    num_decoder_layers=nlayers,
    dim_feedforward=nhid,
    dropout=dropout)

seq_length = 500

data = lmd.load_lmd_runlength(10, seq_length)
data = data[:, :, 1:]
data = torch.tensor(data).float()
data = torch.transpose(data, 0, 1)

model(data[:3])
