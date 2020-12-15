import time
import math
import numpy as np
import torch
import torch.nn as nn
import model_params as params
import copy


class LSTMModel(nn.Module):
    def __init__(self, num_feats, lstm_inp=64, lstm_hidden=200, lstm_layers=2, dim_out=1, dropout=0.1):
        super(LSTMModel, self).__init__()

        self.num_feats = num_feats

        self.lstm_inp = lstm_inp
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.dropout = dropout

        self.embedding_ff = nn.Linear(num_feats, lstm_inp)
        self.gelu = nn.GELU()
        self.lstm = nn.LSTM(lstm_inp, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.decoding_ff_1 = nn.Linear(lstm_hidden * 2, lstm_inp)
        self.decoding_ff_2 = nn.Linear(lstm_inp, dim_out)

    def forward(self, inp):
        # inp = torch.rand(num_seqs, seq_length, num_feats)

        self.lstm.flatten_parameters()

        x = self.gelu(self.embedding_ff(inp))
        x, _ = self.lstm(x)
        x = self.gelu(self.decoding_ff_1(x))
        out = self.decoding_ff_2(x)

        return out


if __name__ == '__main__':
    import plot_outputs as po
    import make_supervised_examples as mse

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from importlib import reload

    reload(mse)
    reload(po)

    seq_length = 200
    num_feats = 3

    lstm_inp = 128
    lstm_hidden = 128
    lstm_layers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(num_feats, lstm_inp, lstm_hidden, lstm_layers).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params} on device {device}')

    num_sequences = 40
    num_epochs = 201

    # inputs = torch.tensor(data[])
    data = torch.rand(num_sequences, seq_length, num_feats).to(device)
    targets = torch.round(torch.rand(num_sequences, seq_length) - 0.45).abs().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=torch.ones(seq_length) * 20).to(device)

    model.train()
    epoch_loss = 0

    for i in range(num_epochs):

        start_time = time.time()
        optimizer.zero_grad()

        output = model(data)
        output_dim = output.shape[-1]

        loss = criterion(output.squeeze(-1), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

        # if not i % 20:
        #     x = (output).detach().cpu().numpy().T
        #     ind = n%rp.random.randint(targets.shape[0])
        #     fig, axs = po.plot(output, targets, ind, num_dur_vals, errored=inputs)
        #     fig.savefig(f'out_imgs/model_test_epoch_{i}.png')
        #     plt.clf()
        #     plt.close(fig)
