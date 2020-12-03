import time
import numpy as np
import torch
import torch.nn as nn
import data_loaders as dl
import factorizations as fcts
import test_trained_model as ttm
import LSTUT_model as lstut
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
reload(lstut)
reload(params)
reload(mse)
reload(ttm)

num_gpus = torch.cuda.device_count()
if torch.cuda.is_available():
    logging.info(f'found {num_gpus} gpus')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.info('defining datasets...')
dset_tr = dl.MidiNoteTupleDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    base='train',
    padding_amt=params.padding_amt,
    trial_run=params.trial_run)
dset_vl = dl.MidiNoteTupleDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    base='validate',
    padding_amt=params.padding_amt,
    trial_run=params.trial_run)
dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
num_feats = dset_tr.num_feats

model = lstut.LSTUTModel(**params.lstut_settings).to(device)
model_size = sum(p.numel() for p in model.parameters())

model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
logging.info(f'created model with n_params={model_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **params.scheduler_settings)

class_ratio = params.seq_length / params.error_indices_settings['num_indices']
criterion = nn.BCEWithLogitsLoss(
    reduction='mean',
    weight=torch.ones(params.seq_length) * class_ratio
    ).to(device)


def prepare_batch(batch):
    inp, target = mse.error_indices(batch, **params.error_indices_settings)
    return inp, target


def log_gpu_info():
    for i in range(torch.cuda.device_count()):
        t = torch.cuda.get_device_properties(i)
        c = torch.cuda.memory_cached(i)
        a = torch.cuda.memory_allocated(i)
        logging.info(f'device: {t}, memory cached: {c}, memory allocated: {a}')


def train_epoch(model, dloader):
    num_seqs_used = 0
    total_loss = 0.

    for i, batch in enumerate(dloader):
        batch = batch.float().to(device)
        input, target = prepare_batch(batch)

        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output.squeeze(-1), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_gradient_norm)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_seqs_used += input.shape[0]
        logging.info(f'    batch {i}, loss {batch_loss:3.7f}')

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
    log_gpu_info()

    # test on validation set
    model.eval()
    num_entries = 0
    val_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(dloader_val):
            batch = batch.float().to(device)
            inp, target = prepare_batch(batch)
            output = model(inp)
            batch_loss = criterion(output.squeeze(-1), target).item()
            val_loss += len(inp) * batch_loss
            num_entries += batch.shape[0]
    val_loss /= num_entries
    val_losses.append(val_loss)

    # keep snapshot of best model
    cur_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # 'dset_stats': dset_tr.get_stats(),
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
        ind = np.random.choice(output.shape[0])
        fig, axs = po.plot_notetuple(inp[ind], output[ind], target[ind])
        fig.savefig(f'./out_imgs/epoch_{epoch}.png', bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    #
    # # save a model checkpoint
    # if (not epoch % params.save_model_every) and epoch > 0 and params.save_model_every > 0:
    #     m_name = (
    #         f'transformer_{params.start_training_time}'
    #         f'_ep-{epoch}_{params.hidden}.{params.d_model}.{params.nlayers}.{params.nhead}.pt')
    #     torch.save(cur_model, m_name)

    # early stopping
    time_since_best = epoch - val_losses.index(min(val_losses))
    if time_since_best > params.early_stopping_patience:
        logging.info(f'stopping early at epoch {epoch}')
        break

# # if max_epochs reached, or early stopping condition reached, save best model
# best_epoch = best_model['epoch']
# m_name = (
#     f'transformer_best_{params.start_training_time}'
#     f'_ep-{best_epoch}_{params.hidden}.{params.d_model}.{params.nlayers}.{params.nhead}.pt')
# torch.save(best_model, m_name)
#
# logging.info('testing best model...')
# dset_tst = dl.MonoFolkSongDataset(
#     dset_fname=params.dset_path,
#     seq_length=params.seq_length,
#     base='test',
#     num_dur_vals=params.num_dur_vals,
#     use_stats=dset_tr.get_stats())
#
# model.load_state_dict(best_model['model_state_dict'])
# res_dict, output = ttm.eval_model(model, dset_tst, device)
#
# with open(params.results_fname, 'w') as f:
#     f.write(ttm.results_string(res_dict, with_params=True, use_duration=dset_tr.use_duration))
