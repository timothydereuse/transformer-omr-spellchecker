import time
import os
import numpy as np
import torch
import torch.nn as nn
import load_meertens_midis as lmm
import factorizations as fcts
import transformer_full_g2p_model as tfgm
import matplotlib.pyplot as plt
from importlib import reload
from torch.utils.data import DataLoader
import plot_outputs as po
reload(lmm)
reload(fcts)
reload(tfgm)

dset_path = r"D:\Documents\MIDI_errors_testing\essen_meertens_songs.hdf5"
val_set_size = 0.1

num_dur_vals = 18
seq_length = 30
batch_size = 500

num_epochs = 200
nhid = 256         # the dimension of the feedforward network
ninp = 256
nlayers = 2        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2          # the number of heads in the multiheadattention models
dropout = 0.1      # the dropout value

lr = 0.002

midi_fnames = lmm.get_all_hdf5_fnames(dset_path)
np.random.shuffle(midi_fnames)
split_pt = int(len(midi_fnames) * val_set_size)
val_fnames = midi_fnames[:split_pt]
train_fnames = midi_fnames[split_pt:]

dset_tr = lmm.MTCDataset(dset_path, seq_length, train_fnames, num_dur_vals)
dset_vl = lmm.MTCDataset(dset_path, seq_length, val_fnames, use_stats_from=dset_tr)
dloader = DataLoader(dset_tr, batch_size)
dloader_val = DataLoader(dset_vl, batch_size)
num_feats = dset_tr.num_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tfgm.TransformerModel(num_feats, ninp, nhid, nlayers, dropout).to(device)
# sum(p.numel() for p in model.parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, threshold=0.001, verbose=True)

# full_loss = nn.BCEWithLogitsLoss(reduction='mean')
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
    dur_loss = dur_criterion(durs, dur_targets_inds)
    return pitch_loss + dur_loss


print('beginning training')
total_loss = 0.
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    num_seqs_used = 0
    for i, batch in enumerate(dloader):
        batch = torch.transpose(batch, 0, 1).float().to(device)
        input, target = (batch, batch)

        optimizer.zero_grad()
        output = model(input, target)
        loss = loss_func(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 2
        num_seqs_used += input.shape[1]
        if i % log_interval == 0:  # and i > 0:
            cur_loss = (total_loss / num_seqs_used)
            elapsed = time.time() - start_time
            # last_lr = scheduler.get_last_lr()[0]
            print(
                f'epoch {epoch:3d} | '
                f'{i:5d} batches | '
                # f'lr {last_lr:02.6f} | '
                f'ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                f'loss {cur_loss:3.5f}')
            total_loss = 0
            num_seqs_used = 0
            start_time = time.time()

    # test on validation set
    model.eval()
    num_entries = 0
    val_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(dloader_val):
            batch = torch.transpose(batch, 0, 1).float().to(device)
            input, target = (batch, batch)
            output = model(input, target)
            val_loss += len(input) * loss_func(output, target).item()
            num_entries += batch.shape[1]
    val_loss /= num_entries

    print(
        f'end of epoch {epoch:3d} | '
        f'val_loss {val_loss:3.5f}')

    scheduler.step(val_loss)

    ind_rand = np.random.choice(output.shape[1])
    fig, axs = po.plot(output, target, ind_rand, num_dur_vals)
    fig.savefig(f'./out_imgs/epoch_{epoch}.png')
    plt.clf()
    plt.close(fig)
