import math
import numpy as np
import torch
import torch.nn as nn

class TransformerMonophonic(nn.Module):

    def __init__(self, num_feats, num_dur_split, nhead, ninp, nhid, nlayers, dropout=0.5):
        super(TransformerMonophonic, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.num_feats = num_feats
        self.ninp = ninp
        self.nhid = nhid

        assert ninp % 2 == 0, "size of embedding must be an even number"
        self.split_embed = ninp // 2

        self.encoder = self.decoder = nn.Sequential(
            nn.Linear(num_feats, self.ninp),
            nn.LeakyReLU()
            )
        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        encoder_layers = TransformerEncoderLayer(self.ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Sequential(
            nn.Linear(self.ninp, self.ninp),
            nn.ReLU()
            )
        self.pitch_decoder = nn.Linear(self.ninp // 2, num_feats - num_dur_split)
        self.dur_decoder = nn.Linear(self.ninp // 2, num_dur_split)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder[0].bias.data.zero_()
        self.pitch_decoder.weight.data.uniform_(-initrange, initrange)
        self.pitch_decoder.bias.data.zero_()
        self.dur_decoder.weight.data.uniform_(-initrange, initrange)
        self.dur_decoder.bias.data.zero_()

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.num_feats)
        src = self.pos_encoder(src)
        trans_output = self.transformer_encoder(src, self.src_mask)
        decode_output = self.decoder(trans_output)

        pitch_output = self.pitch_decoder(decode_output[:, :, self.split_embed:])
        dur_output = self.dur_decoder(decode_output[:, :, :self.split_embed])

        return pitch_output, dur_output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
