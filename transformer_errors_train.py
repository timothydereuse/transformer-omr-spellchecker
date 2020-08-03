import math
import time
import numpy as np
import torch
import torch.nn as nn
import load_lmd as lmd
import factorizations as fcts
import transformer_mono_model as tmm
import matplotlib.pyplot as plt
from importlib import reload
reload(lmd)
reload(fcts)
reload(tmm)


def get_batch(source, i, bptt):
    seq_len = min(bptt, source.shape[1] - 1 - i)
    data = full_set[:-1, i:i + seq_len]
    target = full_set[1:, i:i + seq_len]
    return data, target


# get data from lmd:
mids_path = r"D:\Desktop\meertens_tune_collection\mtc-fs-1.0.tar\midi"
num_dur_vals = 17
delta_mapping, pitch_range = fcts.get_tick_deltas_for_runlength(mids_path, num_dur_vals=17)
full_set = lmd.load_mtc_runlength(delta_mapping, pitch_range, num=200, seq_length=60).float()
pos_weights = lmd.get_class_weights(full_set)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1000
bptt = 100
num_feats = full_set.shape[-1]   # number of features in each token
nhid = 100          # the dimension of the feedforward network model in nn.TransformerEncoder
ninp = 100
nlayers = 3         # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2          # the number of heads in the multiheadattention models
dropout = 0.1      # the dropout value
model = tmm.TransformerMonophonic(num_feats, num_dur_vals, nhead, ninp, nhid, nlayers, dropout).to(device)
# sum(p.numel() for p in model.parameters())

lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)

pitch_criterion = nn.CrossEntropyLoss(reduction='mean', weight=pos_weights[:-num_dur_vals])
dur_criterion = nn.CrossEntropyLoss(reduction='mean', weight=pos_weights[-num_dur_vals:])


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
start_time = time.time()
for epoch in range(num_epochs):
    for batch, i in enumerate(range(0, full_set.shape[1] - 1, bptt)):
        data, target = get_batch(full_set, i, bptt)
        optimizer.zero_grad()
        pitch_output, dur_output = model(data)
        loss = loss_func(pitch_output, dur_output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 5
        if batch % log_interval == 0: #and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    epoch, batch, full_set.shape[1] // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
    scheduler.step()
