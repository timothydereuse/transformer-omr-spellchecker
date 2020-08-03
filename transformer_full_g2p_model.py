import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_feats, hidden=100, feedforward_size=100, nlayers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        nhead = math.ceil(hidden/64)

        self.encoder = nn.Sequential(
            nn.Linear(num_feats, hidden),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Sequential(
            nn.Linear(num_feats, hidden),
            nn.ReLU(),
        )
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation='relu'
            )

        self.fc_out = nn.Linear(hidden, num_feats)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self._generate_point_mask(len(trg)).to(trg.device)
            # self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
        if self.src_mask is None or self.src_mask.size(0) != len(trg):
            self.src_mask = self._generate_point_mask(len(src)).to(src.device)
            # self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        # src_pad_mask = self.make_len_mask(src)
        # trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(
            src, trg,
            src_mask=self.src_mask,
            tgt_mask=self.trg_mask,
            memory_mask=self.memory_mask,
            # src_key_padding_mask=src_pad_mask,
            # tgt_key_padding_mask=trg_pad_mask,
            # memory_key_padding_mask=src_pad_mask
        )
        output = self.fc_out(output)

        return output

    def _generate_point_mask(self, sz):
        mask = torch.diag(torch.ones(sz))
        # mask = torch.zeros(sz, sz)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':
    from itertools import product
    import matplotlib.pyplot as plt
    seq_length = 30
    num_seqs = 50
    num_feats = 30
    num_dur_vals = 10

    data_r = torch.rand(seq_length, num_seqs, num_feats)
    data = torch.zeros_like(data_r)

    note_inds = torch.max(data_r[:, :, :-num_dur_vals], 2).indices
    dur_inds = torch.max(data_r[:, :, -num_dur_vals:], 2).indices

    # i have no idea how to vectorize this.
    for i, j in product(range(seq_length), range(num_seqs)):
        ss = np.sqrt(2)
        ind = int((j + i * ss) % (num_feats - num_dur_vals))
        data[i][j][ind] = 1
    for i, j in product(range(seq_length), range(num_seqs)):
        ss = np.sqrt(3)
        ind = int((i + j * ss) % (num_dur_vals) + (num_feats - num_dur_vals))
        data[i][j][ind] = 1

    # inputs = torch.tensor(data[])
    inputs = data
    targets = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 500
    hidden = 60
    feedforward = 60
    nlayers = 3        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2          # the number of heads in the multiheadattention models
    dropout = 0.1      # the dropout value
    model = TransformerModel(num_feats, hidden, feedforward, nlayers, dropout).to(device)
    print(sum(p.numel() for p in model.parameters()))

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    full_loss = nn.BCEWithLogitsLoss(reduction='mean')

    pitch_criterion = nn.CrossEntropyLoss(reduction='mean')
    dur_criterion = nn.CrossEntropyLoss(reduction='mean')

    def loss_func(outputs, targets):

        pitch_targets = targets[:, :, :-num_dur_vals]
        dur_targets = targets[:, :, -num_dur_vals:]

        pitches = outputs[:, :, :-num_dur_vals]
        pitches = pitches.view(-1, pitches.shape[-1])
        durs = outputs[:, :, -num_dur_vals:]
        durs = durs.view(-1, durs.shape[-1])

        pitch_targets_inds = pitch_targets.reshape(-1, pitch_targets.shape[-1]).max(1).indices
        dur_targets_inds = dur_targets.reshape(-1, num_dur_vals).max(1).indices

        pitch_loss = pitch_criterion(pitches, pitch_targets_inds)
        dur_loss = dur_criterion(durs.view(-1, num_dur_vals), dur_targets_inds)
        return pitch_loss + dur_loss

    model.train()
    epoch_loss = 0

    for i in range(num_epochs):
        optimizer.zero_grad()

        output = model(inputs, targets)
        output_dim = output.shape[-1]

        # loss = full_loss(output.view(-1, output_dim), targets.view(-1, output_dim))
        loss = loss_func(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"epoch: {i} | loss: {loss.item()}")

    x = (output).detach().cpu().numpy().T
    plt.imshow(x[:, 10])
    plt.show()
