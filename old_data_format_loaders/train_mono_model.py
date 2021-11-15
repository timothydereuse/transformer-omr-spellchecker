import time
import numpy as np
import torch
import torch.nn as nn
import mono_folk_song_dataloader as dl
import factorizations as fcts
import test_trained_model as ttm
import transformer_encoder_model as tem
import make_supervised_examples as mse
from importlib import reload
from torch.utils.data import DataLoader
import plot_outputs as po
import model_params as params
import logging
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reload(dl)
reload(fcts)
reload(tem)
reload(params)
reload(mse)
reload(ttm)

logging.info('defining datasets...')
dset_tr = dl.MonoFolkSongDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    num_dur_vals=params.num_dur_vals,
    base='train',
    proportion_for_stats=params.proportion_for_stats,
    padding_amt=params.padding_amt,
    trial_run=params.trial_run)
dset_vl = dl.MonoFolkSongDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    num_dur_vals=params.num_dur_vals,
    base='validate',
    use_stats=dset_tr.get_stats(),
    padding_amt=params.padding_amt,
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


def loss_func(outputs, targets, pitch_criterion, dur_criterion):
    ndv = dset_tr.dur_subvector_len

    pitch_targets = targets[:, :, :-ndv] if ndv > 0 else targets
    pitches = outputs[:, :, :-ndv] if ndv > 0 else outputs
    pitches = pitches.view(-1, pitches.shape[-1])
    pitch_targets_inds = pitch_targets.reshape(-1, pitch_targets.shape[-1]).max(1).indices
    pitch_loss = pitch_criterion(pitches, pitch_targets_inds)

    if ndv > 0:
        dur_targets = targets[:, :, -ndv:]
        durs = outputs[:, :, -ndv:]
        durs = durs.view(-1, durs.shape[-1])
        dur_targets_inds = dur_targets.reshape(-1, ndv).max(1).indices
        dur_loss = dur_criterion(durs, dur_targets_inds)
        res = pitch_loss + dur_loss
    else:
        res = pitch_loss

    return res


def prepare_batch(batch):
    # input, _ = mse.remove_indices(batch, **params.remove_indices_settings)
    inp, _ = mse.mask_indices(batch, **params.mask_indices_settings)
    target = batch
    return inp, target


def train_epoch(model, dloader):
    num_seqs_used = 0
    total_loss = 0.

    for i, batch in enumerate(dloader):
        batch = batch.float().to(device)
        input, target = prepare_batch(batch)

        optimizer.zero_grad()
        output = model(input)

        loss = loss_func(output, target, pitch_criterion, dur_criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_gradient_norm)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_seqs_used += input.shape[0]

    mean_loss = total_loss / num_seqs_used
    return mean_loss


logging.info('beginning training')
start_time = time.time()
val_losses = []
best_model = None
for epoch in range(params.num_epochs):

    # perform training epoch
    model.train()
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
            batch_loss = loss_func(output, target, pitch_criterion, dur_criterion).item()
            val_loss += len(input) * batch_loss
            num_entries += batch.shape[0]
    val_loss /= num_entries
    val_losses.append(val_loss)

    # keep snapshot of best model
    cur_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'dset_stats': dset_tr.get_stats(),
            'val_losses': val_losses
            }
    if len(val_losses) == 1 or val_losses[-1] < min(val_losses[:-1]):
        best_model = copy.deepcopy(cur_model)

    elapsed = time.time() - start_time
    logging.info(
        f'epoch {epoch:3d} | '
        f's/epoch {elapsed:3.5f} | '
        f'train_loss {cur_loss:3.7f} | '
        f'val_loss {val_loss:3.7f} ')
    start_time = time.time()

    scheduler.step(val_loss)

    # save an image
    if not epoch % params.save_img_every and epoch > 0:
        ind_rand = np.random.choice(output.shape[0])
        fig, axs = po.plot(output, target, ind_rand, dset_tr.dur_subvector_len, errored=input)
        fig.savefig(f'./out_imgs/epoch_{epoch}.png', bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    # save a model checkpoint
    if (not epoch % params.save_model_every) and epoch > 0 and params.save_model_every > 0:
        m_name = (
            f'transformer_{params.start_training_time}'
            f'_ep-{epoch}_{params.hidden}.{params.d_model}.{params.nlayers}.{params.nhead}.pt')
        torch.save(cur_model, m_name)

    # early stopping
    time_since_best = epoch - val_losses.index(min(val_losses))
    if time_since_best > params.early_stopping_patience:
        logging.info(f'stopping early at epoch {epoch}')
        break

# if max_epochs reached, or early stopping condition reached, save best model
best_epoch = best_model['epoch']
m_name = (
    f'transformer_best_{params.start_training_time}'
    f'_ep-{best_epoch}_{params.hidden}.{params.d_model}.{params.nlayers}.{params.nhead}.pt')
torch.save(best_model, m_name)

logging.info('testing best model...')
dset_tst = dl.MonoFolkSongDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    base='test',
    num_dur_vals=params.num_dur_vals,
    use_stats=dset_tr.get_stats())

model.load_state_dict(best_model['model_state_dict'])
res_dict, output = ttm.eval_model(model, dset_tst, device)

with open(params.results_fname, 'w') as f:
    f.write(ttm.results_string(res_dict, with_params=True, use_duration=dset_tr.use_duration))
