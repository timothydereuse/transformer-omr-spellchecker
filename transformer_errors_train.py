import time
import numpy as np
import torch
import torch.nn as nn
import data_loaders as dl
import factorizations as fcts
import transformer_full_seq_model as tfgm
from importlib import reload
from torch.utils.data import DataLoader
import plot_outputs as po

print('matplotlib...')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('reloads...')
reload(dl)
reload(fcts)
reload(tfgm)

print('definitions...')
dset_path = r"essen_meertens_songs.hdf5"
val_set_size = 0.1

num_dur_vals = 15
seq_length = 50
batch_size = 700

num_epochs = 150
nhid = 256 // 4        # the dimension of the feedforward network
ninp = 256 // 4
nlayers = 2        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2          # the number of heads in the multiheadattention models
dropout = 0.1      # the dropout value

lr = 0.003

print('reading hdf5...')
midi_fnames = dl.get_all_hdf5_fnames(dset_path)
np.random.shuffle(midi_fnames)
split_pt = int(len(midi_fnames) * val_set_size)
val_fnames = midi_fnames[:split_pt]
train_fnames = midi_fnames[split_pt:]

print('defining datasets...')
dset_tr = dl.MonoFolkSongDataset(dset_path, seq_length, train_fnames, num_dur_vals)
dset_vl = dl.MonoFolkSongDataset(dset_path, seq_length, val_fnames, use_stats_from=dset_tr)
dloader = DataLoader(dset_tr, batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, batch_size, pin_memory=True)
num_feats = dset_tr.num_feats

print('creating model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device selected: {device}')
model = tfgm.TransformerModel(num_feats, ninp, nhid, nlayers, dropout).to(device)
model_size = sum(p.numel() for p in model.parameters())
print(f'instantiated model with nparams={model_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, threshold=0.001, verbose=True)

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
    dur_loss = dur_criterion(durs, dur_targets_inds)
    return pitch_loss + dur_loss


def train_epoch(model, dloader):
    num_seqs_used = 0
    total_loss = 0.
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
        num_seqs_used += input.shape[1]

        print(f'batch {i}')

    mean_loss = total_loss / num_seqs_used
    return mean_loss


print('beginning training')
start_time = time.time()
for epoch in range(num_epochs):
    model.train()

    # perform training epoch
    cur_loss = train_epoch(model, dloader)

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

    elapsed = time.time() - start_time
    print(
        f'epoch {epoch:3d} | '
        f'ms/epoch {elapsed * 1000:5.2f} | '
        f'train_loss {cur_loss:3.5f} | '
        f'val_loss {val_loss:3.5f} ')
    start_time = time.time()

    scheduler.step(val_loss)

    save_every = 15
    if not epoch % save_every:
        ind_rand = np.random.choice(output.shape[1])
        fig, axs = po.plot(output, target, ind_rand, num_dur_vals)
        fig.savefig(f'./out_imgs/epoch_{epoch}.png')
        plt.clf()
        plt.close(fig)

        m_name = f'transformer_epoch-{epoch}_{nhid}.{ninp}.{nlayers}.{nhead}.pt'
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'loss': loss,
                }, m_name)
