import time
import numpy as np
import torch
import torch.nn as nn
# import data_loaders as dl
import data_management.toy_datasets as dl
import factorizations as fcts
import test_trained_model as ttm
import models.transformer_autoregressive_model as tam
import test_trained_notetuple_model as ttnm
import training_helper_functions as tr_funcs
from importlib import reload
from torch.utils.data import DataLoader
import plot_outputs as po
import model_params as params
import logging
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reload(tr_funcs)
reload(ttnm)
reload(dl)
reload(fcts)
reload(tam)
reload(params)
reload(ttm)
reload(po)

num_gpus = torch.cuda.device_count()
if torch.cuda.is_available():
    logging.info(f'found {num_gpus} gpus')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.info('defining datasets...')
dset_tr = dl.SequenceCopyDataset(
    num_feats=params.transformer_ar_settings['input_feats'],
    num_seqs=1000,
    seq_length=100,
    seq_period=24,
    freq_vary=0.4)
dset_vl = dl.SequenceCopyDataset(
    num_feats=params.transformer_ar_settings['input_feats'],
    num_seqs=100,
    seq_length=100,
    seq_period=24,
    freq_vary=0.4)
dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
num_feats = dset_tr.num_feats

model = tam.TransformerEncoderDecoder(**params.transformer_ar_settings).to(device)
model_size = sum(p.numel() for p in model.parameters())

# model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
logging.info(f'created model with n_params={model_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **params.scheduler_settings)

criterion = nn.MSELoss(reduction='sum').to(device)

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
    logging.info(
        f'epoch {epoch:3d} | '
        f's/epoch {elapsed:3.5f} | '
        f'train_loss {train_loss:3.7f} | '
        f'val_loss {val_loss:3.7f} ')
    start_time = time.time()

    scheduler.step(val_loss)

    # save an image
    if not epoch % params.save_img_every and epoch > 0:

        inference_test = val_exs[0]
        res = model.inference_decode(inference_test, 25)
        inferred = torch.cat((inference_test, res), 1).detach().numpy()

        ind = np.random.choice(val_exs[0].shape[0])
        # fig, axs = po.plot_notetuple(inp[ind], output[ind], target[ind], F1_thresh)
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        for ft in range(val_exs[0].shape[-1]):
            axs[0].plot(val_exs[2][ind, :, ft])
            axs[1].plot(val_exs[1][ind, :, ft])
            axs[2].plot(inferred[ind, :, ft])
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
