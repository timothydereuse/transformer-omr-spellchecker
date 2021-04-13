import time
import numpy as np
import torch
import torch.nn as nn
import point_set_dataloader as dl
import factorizations as fcts
import models.set_transformer_model as stm
import training_helper_functions as tr_funcs
from torch.utils.data import DataLoader
from chamferdist import ChamferDistance
import plot_outputs as po
import model_params as params
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib import reload
reload(tr_funcs)
reload(dl)
reload(fcts)
reload(stm)
reload(params)
reload(po)

device, num_gpus = tr_funcs.get_cuda_info()

logging.info('defining datasets...')
dset_tr = dl.MidiNoteTupleDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    base='train',
    num_feats=params.num_feats,
    padding_amt=params.padding_amt,
    trial_run=params.trial_run)
dset_vl = dl.MidiNoteTupleDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    base='validate',
    num_feats=params.num_feats,
    padding_amt=params.padding_amt,
    trial_run=params.trial_run,
    use_stats_from=dset_tr)

dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
num_feats = dset_tr.num_feats

model = stm.SetTransformer(**params.set_transformer_settings).to(device)
model_size = sum(p.numel() for p in model.parameters())

model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
logging.info(f'created model with n_params={model_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **params.scheduler_settings)

cd = ChamferDistance().to(device)
criterion = lambda x, y: cd(x, y, bidirectional=True)

logging.info('beginning training')
tr_funcs.log_gpu_info()
start_time = time.time()
val_losses = []
best_model = None
for epoch in range(params.num_epochs):

    # perform training epoch
    model.train()
    train_loss, tr_exs = tr_funcs.run_epoch(model, dloader, optimizer,
                                            criterion, device, train=True, log_each_batch=True)

    # test on validation set
    model.eval()
    num_entries = 0
    val_loss = 0.
    with torch.no_grad():
        val_loss, val_exs = tr_funcs.run_epoch(model, dloader, optimizer,
                                               criterion, device, train=False, log_each_batch=False)

    val_losses.append(val_loss)

    elapsed = time.time() - start_time
    log_train_loss = np.log(train_loss)
    log_val_loss = np.log(val_loss)
    logging.info(
        f'epoch {epoch:3d} | '
        f's/epoch {elapsed:3.5f} | '
        f'train_loss {log_train_loss:2.7f} | '
        f'val_loss {log_val_loss:2.7f} ')
    start_time = time.time()

    scheduler.step(val_loss)

    # save an image
    if not epoch % params.save_img_every and epoch > 0:

        ind = np.random.choice(tr_exs['input'].shape[0])
        fig, axs = po.plot_set(tr_exs, dset_tr, ind)
        fig.savefig(f'./out_imgs/epoch_{epoch}.png', bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    # save a model checkpoint
    # if (not epoch % params.save_model_every) and epoch > 0 and params.save_model_every > 0:
    #     m_name = (
    #         f'lstut_{params.start_training_time}'
    #         f'_{params.lstut_summary_str}.pt')
    #     torch.save(cur_model, m_name)

    # # keep snapshot of best model
    # cur_model = {
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict(),
    #         'val_losses': val_losses,
    #         }
    # if len(val_losses) == 1 or val_losses[-1] < min(val_losses[:-1]):
    #     best_model = copy.deepcopy(cur_model)

    # early stopping
    time_since_best = epoch - val_losses.index(min(val_losses))
    if time_since_best > params.early_stopping_patience:
        logging.info(f'stopping early at epoch {epoch}')
        break

# # if max_epochs reached, or early stopping condition reached, save best model
# best_epoch = best_model['epoch']
# m_name = (f'lstut_best_{params.start_training_time}_{params.lstut_summary_str}.pt')
# torch.save(best_model, m_name)
