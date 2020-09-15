import time
import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=150):
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

        self.fc_out = nn.Sequential(
            nn.Linear(hidden, num_feats),
        )

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        # data points are padding iff the first and last elements of the feature vector are 1
        res = inp[:, :, 0] * inp[:, :, -1]
        return res.transpose(0, 1).to(torch.bool)

    def forward(self, src, trg):
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(trg):
            self.tgt_mask = None
            # self.tgt_mask = self._generate_point_mask(len(trg)).to(trg.device)
            # self.tgt_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
        if self.src_mask is None or self.src_mask.size(0) != len(trg):
            self.src_mask = None
            # self.src_mask = self._generate_point_mask(len(src)).to(src.device)
            # self.tgt_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(
            src, trg,
            src_mask=self.src_mask,
            tgt_mask=self.tgt_mask,
            memory_mask=self.memory_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask
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
    import plot_outputs

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seq_length = 50
    num_seqs = 1000
    num_feats = 50
    num_dur_vals = 10

    num_epochs = 140
    hidden = 100
    feedforward = 100
    nlayers = 3        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2          # the number of heads in the multiheadattention models
    dropout = 0.1      # the dropout value

    data_r = torch.rand(seq_length, num_seqs, num_feats)
    data = torch.zeros_like(data_r)

    note_inds = torch.max(data_r[:, :, :-num_dur_vals], 2).indices
    dur_inds = torch.max(data_r[:, :, -num_dur_vals:], 2).indices

    # i have no idea how to vectorize this.
    for i, j in product(range(seq_length), range(num_seqs)):
        ss = np.sqrt(3)
        ind = int((j * i * ss) % (num_feats - num_dur_vals))
        data[i][j][ind] = 1
    for i, j in product(range(seq_length), range(num_seqs)):
        ss = np.sqrt(2)
        ind = int((i + j * ss) % (num_dur_vals) + (num_feats - num_dur_vals))
        data[i][j][ind] = 1

    # inputs = torch.tensor(data[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = data.to(device)
    targets = data.to(device)

    model = TransformerModel(num_feats, hidden, feedforward, nlayers, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params} on device {device}')

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    full_loss = nn.BCEWithLogitsLoss(reduction='mean')

    pitch_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    dur_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

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

        start_time = time.time()
        optimizer.zero_grad()

        output = model(inputs, targets)
        output_dim = output.shape[-1]

        # loss = full_loss(output.view(-1, output_dim), targets.view(-1, output_dim))
        loss = loss_func(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

        if not i % 25:
            x = (output).detach().cpu().numpy().T
            fig, axs = plot_outputs.plot(output, targets, 3, num_dur_vals)
            fig.savefig(f'model_test_epoch_{i}.png')
            plt.clf()
            plt.close(fig)
