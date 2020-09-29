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


class TransformerBidirectionalModel(nn.Module):
    def __init__(self, num_feats, d_model=100, hidden=100, nlayers=3, nhead=2, dropout=0.1):
        super(TransformerBidirectionalModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_feats, d_model),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, num_feats),
        )

        self.src_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        # data points are padding iff the first and last elements of the feature vector are 1
        res = inp[:, :, 0] * inp[:, :, -1]
        return res.to(torch.bool)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
            self.src_mask = None
            self.src_mask = self._generate_point_mask(src.shape[1]).to(src.device)

        src_pad_mask = self.make_len_mask(src)

        src = self.encoder(src)

        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src,
            mask=self.src_mask,
            src_key_padding_mask=src_pad_mask,
        )
        output = output.transpose(0, 1)

        output = self.fc_out(output)
        return output

    def _generate_point_mask(self, sz):
        mask = torch.diag(torch.ones(sz))
        # mask = torch.zeros(sz, sz)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':
    from itertools import product
    import plot_outputs as po
    import make_supervised_examples as mse

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from importlib import reload

    reload(mse)
    reload(po)

    seq_length = 30
    num_seqs = 300
    num_feats = 40
    num_dur_vals = 10
    mask_inds_num = 4

    num_epochs = 201
    d_model = 128
    hidden = 256
    nlayers = 2        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2          # the number of heads in the multiheadattention models
    dropout = 0.1      # the dropout value

    lr = 0.0005

    data_r = torch.rand(num_seqs, seq_length, num_feats)
    data = torch.zeros_like(data_r)

    note_inds = torch.max(data_r[:, :, :-num_dur_vals], 2).indices
    dur_inds = torch.max(data_r[:, :, -num_dur_vals:], 2).indices

    # i have no idea how to vectorize this.
    for i, j in product(range(seq_length), range(num_seqs)):
        ss = np.sqrt(3)
        ind = int((j * i * ss) % (num_feats - num_dur_vals))
        data[j][i][ind] = 1
    for i, j in product(range(seq_length), range(num_seqs)):
        ss = np.sqrt(2)
        ind = int((i + j * ss) % (num_dur_vals) + (num_feats - num_dur_vals))
        data[j][i][ind] = 1

    # inputs = torch.tensor(data[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = data.to(device)
    inputs, inds = mse.mask_indices(data, mask_inds_num)
    inputs = inputs.to(device)

    model = TransformerBidirectionalModel(
        num_feats, d_model, hidden, nlayers, nhead, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params} on device {device}')

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

        output = model(inputs)
        output_dim = output.shape[-1]

        # loss = full_loss(output.view(-1, output_dim), targets.view(-1, output_dim))
        loss = loss_func(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

        if not i % 20:
            x = (output).detach().cpu().numpy().T
            ind = np.random.randint(targets.shape[0])
            fig, axs = po.plot(output, targets, ind, num_dur_vals, errored=inputs)
            fig.savefig(f'out_imgs/model_test_epoch_{i}.png')
            plt.clf()
            plt.close(fig)
