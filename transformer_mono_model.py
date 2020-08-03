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


if __name__ == '__main__':
    from itertools import product
    import plot_outputs as po
    seq_length = 20
    num_seqs = 20
    num_feats = 20
    num_dur_vals = 10

    data_r = torch.rand(seq_length, num_seqs, num_feats)
    data = torch.zeros_like(data_r)

    note_inds = torch.max(data_r[:, :, :-num_dur_vals], 2).indices
    dur_inds = torch.max(data_r[:, :, -num_dur_vals:], 2).indices

    # i have no idea how to vectorize this.
    for i, j in product(range(seq_length), range(num_seqs)):
        data[i][j][note_inds[i][j]] = 1
    for i, j in product(range(seq_length), range(num_seqs)):
        data[i][j][dur_inds[i][j] + num_feats - num_dur_vals] = 1

    inputs = data[:-1]
    targets = data[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 300
    nhid = 100         # the dimension of the feedforward network model in nn.TransformerEncoder
    ninp = 100
    nlayers = 3        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2          # the number of heads in the multiheadattention models
    dropout = 0.1      # the dropout value
    model = TransformerMonophonic(num_feats, num_dur_vals, nhead, ninp, nhid, nlayers, dropout).to(device)
    print(sum(p.numel() for p in model.parameters()))

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pitch_criterion = nn.CrossEntropyLoss(reduction='mean')
    dur_criterion = nn.CrossEntropyLoss(reduction='mean')

    def loss_func(pitches, durs, targets):

        pitch_targets = targets[:, :, :-num_dur_vals]
        dur_targets = targets[:, :, -num_dur_vals:]

        pitch_targets_inds = pitch_targets.reshape(-1, pitch_targets.shape[-1]).max(1).indices
        dur_targets_inds = dur_targets.reshape(-1, num_dur_vals).max(1).indices

        pitch_loss = pitch_criterion(pitches.view(-1, pitches.shape[-1]), pitch_targets_inds)
        dur_loss = dur_criterion(durs.view(-1, num_dur_vals), dur_targets_inds)
        return pitch_loss + dur_loss

    # TRAINING LOOP
    model.train()
    total_loss = 0.
    for batch in range(num_epochs):
        optimizer.zero_grad()
        pitch_output, dur_output = model(inputs)
        loss = loss_func(pitch_output, dur_output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 5
        if batch % log_interval == 0:
            cur_loss = total_loss / log_interval
            print(' batch {:3d} | loss {:5.2f}'.format(batch, cur_loss))
            total_loss = 0
