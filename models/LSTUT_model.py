import time
import math
import numpy as np
import torch
import torch.nn as nn
import model_params as params
from linformer_pytorch import Linformer
import copy


class LSTUTModel(nn.Module):
    def __init__(self, seq_length, num_feats, lstm_inp=64, lstm_hidden=200, lstm_layers=2, tf_inp=64, tf_hidden=128, tf_depth=4, tf_k=128, nhead=4, dim_out=1, dropout=0.1):
        super(LSTUTModel, self).__init__()

        self.seq_length = seq_length
        self.num_feats = num_feats

        self.lstm_inp = lstm_inp
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.tf_inp = tf_inp
        self.tf_hidden = tf_hidden
        self.tf_k = tf_k
        self.dim_out = dim_out

        self.nhead = nhead
        self.tf_depth = tf_depth
        self.dropout = dropout

        # n.b. LSTM takes on the role of positional encoding - no need for it!
        self.embedding_ff = nn.Linear(num_feats, lstm_inp)
        self.gelu = nn.GELU()
        # self.lstm1 = nn.LSTM(lstm_inp, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True)
        # self.lstm_to_tf_ff = nn.Linear(lstm_hidden * 2, tf_inp)
        self.lstm1 = nn.LSTM(lstm_inp, tf_inp, lstm_layers, batch_first=True, bidirectional=False)
        # self.lstm_to_tf_ff = nn.Linear(lstm_hidden * 2, tf_inp)
        self.transformer = Linformer(
            input_size=seq_length,
            channels=tf_inp,
            dim_k=tf_k,
            dim_ff=tf_hidden,
            dropout=dropout,
            nhead=nhead,
            depth=tf_depth,
            parameter_sharing="layerwise",
            )
        self.norm = nn.LayerNorm(tf_inp)
        self.lstm2 = nn.LSTM(tf_inp, lstm_hidden, lstm_layers, batch_first=True, bidirectional=False)
        self.output_ff = nn.Linear(lstm_hidden, dim_out)

    def forward(self, inp):
        # inp = torch.rand(num_seqs, seq_length, num_feats)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        x = self.gelu(self.embedding_ff(inp))
        lstm_out, _ = self.lstm1(x)

        # lstm_out = self.gelu(self.lstm_to_tf_ff(x))
        x = lstm_out.clone()
        x = self.transformer(x)
        x = self.norm(x + lstm_out)
        x, _ = self.lstm2(x)
        out = self.output_ff(x)



        return out

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

    seq_length = 20
    num_feats = 3

    lstm_inp = 6
    lstm_hidden = 6
    lstm_layers = 1
    tf_inp = 128
    tf_hidden = 128
    tf_k = 128

    nhead = 4
    tf_depth = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTUTModel(
        seq_length, num_feats, lstm_inp, lstm_hidden, lstm_layers, tf_inp, tf_hidden, tf_depth, tf_k, nhead
    ).to(device)

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
