import time
import numpy as np
import torch
import torch.nn as nn
import data_loaders as dl
import factorizations as fcts
import transformer_encoder_model as tem
import make_supervised_examples as mse
from importlib import reload
from torch.utils.data import DataLoader
import plot_outputs as po
import model_params as params
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reload(dl)
reload(fcts)
reload(tem)
reload(params)
reload(mse)

logging.info('defining datasets...')
dset_tr = dl.MonoFolkSongDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    num_dur_vals=params.num_dur_vals,
    base='train',
    proportion_for_stats=params.proportion_for_stats,
    trial_run=params.trial_run)
dset_vl = dl.MonoFolkSongDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    num_dur_vals=params.num_dur_vals,
    base='validate',
    use_stats=dset_tr.get_stats(),
    trial_run=params.trial_run)
dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
num_feats = dset_tr.num_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tem.TransformerBidirectionalModel(
        num_feats=num_feats,
        d_model=params.d_model,
        hidden=params.hidden,
        nlayers=params.nlayers,
        nhead=params.nhead,
        dropout=params.dropout
        ).to(device)
model_size = sum(p.numel() for p in model.parameters())
logging.info(f'created model with n_params={model_size} on device {device}')

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=params.lr_plateau_factor,
            patience=params.lr_plateau_patience,
            threshold=params.lr_plateau_threshold,
            verbose=True)

pitch_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
dur_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)


def loss_func(outputs, targets):
    ndv = dset_tr.dur_subvector_len
    pitch_targets = targets[:, :, :-ndv]
    dur_targets = targets[:, :, -ndv:]

    pitches = outputs[:, :, :-ndv]
    pitches = pitches.view(-1, pitches.shape[-1])
    durs = outputs[:, :, -ndv:]
    durs = durs.view(-1, durs.shape[-1])

    pitch_targets_inds = pitch_targets.reshape(-1, pitch_targets.shape[-1]).max(1).indices
    dur_targets_inds = dur_targets.reshape(-1, ndv).max(1).indices

    pitch_loss = pitch_criterion(pitches, pitch_targets_inds)
    dur_loss = dur_criterion(durs, dur_targets_inds)
    return pitch_loss + dur_loss


def prepare_batch(batch):
    # input, _ = mse.remove_indices(batch, **params.remove_indices_settings)
    input, _ = mse.mask_indices(batch, **params.mask_indices_settings)
    input = input.transpose(1, 0)
    target = batch.transpose(1, 0)
    return input, target
    # return batch.transpose(0, 1), batch.transpose(0, 1)


def train_epoch(model, dloader):
    num_seqs_used = 0
    total_loss = 0.

    for i, batch in enumerate(dloader):
        batch = batch.float().to(device)
        input, target = prepare_batch(batch)

        optimizer.zero_grad()
        output = model(input)

        loss = loss_func(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_gradient_norm)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_seqs_used += input.shape[1]
        # logging.info(f'batch {i} | loss {batch_loss}')

    mean_loss = total_loss / num_seqs_used
    return mean_loss


logging.info('beginning training')
start_time = time.time()
val_losses = []
best_model = None
for epoch in range(params.num_epochs):
    model.train()

    # perform training epoch
    cur_loss = train_epoch(model, dloader)

    # test on validation set
    model.eval()
    num_entries = 0
    val_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(dloader_val):
            batch = batch.float().to(device)
            input, target = prepare_batch(batch)
            output = model(input)
            val_loss += len(input) * loss_func(output, target).item()
            num_entries += batch.shape[1]
    val_loss /= num_entries
    val_losses.append(val_loss)

    cur_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'dset_stats': dset_tr.get_stats(),
            'val_losses': val_losses
            }

    if val_loss < min(val_losses):
        best_model = cur_model

    elapsed = time.time() - start_time
    logging.info(
        f'epoch {epoch:3d} | '
        f's/epoch {elapsed:3.5f} | '
        f'train_loss {cur_loss:3.7f} | '
        f'val_loss {val_loss:3.7f} ')
    start_time = time.time()

    scheduler.step(val_loss)

    if not params.trial_run and not epoch % params.save_img_every and epoch > 0:
        ind_rand = np.random.choice(output.shape[1])
        fig, axs = po.plot(output, target, ind_rand, params.num_dur_vals, errored=input)
        fig.savefig(f'./out_imgs/epoch_{epoch}.png')
        plt.clf()
        plt.close(fig)

    if not params.trial_run and not epoch % params.save_model_every and epoch > 0:
        m_name = (
            f'transformer_{params.start_training_time}'
            f'_ep-{epoch}_{params.hidden}.{params.d_model}.{params.nlayers}.{params.nhead}.pt')
        torch.save(cur_model, m_name)

m_name = (
    f'transformer_best_{params.start_training_time}'
    f'_ep-{epoch}_{params.hidden}.{params.d_model}.{params.nlayers}.{params.nhead}.pt')
torch.save(best_model, m_name)
