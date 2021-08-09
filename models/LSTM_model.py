import time
import math
import numpy as np
import torch
import torch.nn as nn
import model_params as params
import copy


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, seq_length, lstm_inp=64, lstm_hidden=200, lstm_layers=2, dim_out=1, dropout=0.1):
        super(LSTMModel, self).__init__()

        self.vocab_size = vocab_size

        self.lstm_inp = lstm_inp
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.seq_length = seq_length

        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.lstm_inp, padding_idx=1)
        self.gelu = nn.GELU()
        self.lstm = nn.LSTM(lstm_inp, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.bn = nn.BatchNorm1d(seq_length)
        self.decoding_ff_1 = nn.Linear(lstm_hidden * 2, dim_out)
        self.sig = nn.Sigmoid()

    def forward(self, inp):
        # inp = torch.rand(num_seqs, seq_length, num_feats)

        self.lstm.flatten_parameters()

        x = self.embedding(inp)
        x, _ = self.lstm(x)
        x = self.bn(x)
        # out = self.sig(self.decoding_ff_1(x))
        out = self.decoding_ff_1(x)


        return out


if __name__ == '__main__':

    batch_size = 10
    seq_len = 256
    output_pts = 1
    num_feats = 1
    vocab_size = 100

    X = (torch.rand(batch_size, seq_len) * vocab_size).floor().type(torch.long)
    tgt = (torch.rand(batch_size, seq_len, output_pts) - 0.00005).round()

    model = LSTMModel(
        vocab_size=vocab_size,
        seq_length=seq_len,
        lstm_inp=128,
        lstm_hidden=64,
        lstm_layers=1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    res = model(X)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.BCELoss()
    sched = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.25, verbose=False)

    num_epochs = 1000

    model.train()
    epoch_loss = 0
    for i in range(num_epochs):

        start_time = time.time()
        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, tgt)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        sched.step()

        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

    model.eval()
