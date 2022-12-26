import time
import numpy as np
import torch
import torch.nn as nn
import agnostic_omr_dataloader as dl
import test_results_metrics as ttnm
import models.LSTM_model as lstm
import data_augmentation.error_gen_logistic_regression as err_gen
import training_helper_functions as tr_funcs
import data_management.vocabulary as vocab
from torch.utils.data import DataLoader
import plot_outputs as po
import model_params
import logging
import argparse
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib import reload
reload(tr_funcs)
reload(dl)
reload(lstm)
reload(model_params)
reload(po)
reload(err_gen)
reload(ttnm)

parser = argparse.ArgumentParser(description='Training script, with optional parameter searching.')
parser.add_argument('parameters', default='default_params.json',
                    help='Parameter file in .json format.')
parser.add_argument('-m', '--mod_number', type=int, default=0,
                    help='Index of specific modification to apply to given parameter set.')
parser.add_argument('-l', '--logging', action='store_true',
                    help='Whether or not to log training results to file.')
parser.add_argument('-d', '--dryrun', action='store_true',
                    help='Halts execution immediately before training begins.')
args = vars(parser.parse_args())

params = model_params.Params(args['parameters'],  args['logging'], args['mod_number'])
dry_run = args['dryrun']

device, num_gpus = tr_funcs.get_cuda_info()
logging.info('defining datasets...')

v = vocab.Vocabulary(load_from_file=params.saved_vocabulary)
error_generator = err_gen.ErrorGenerator(
    ngram=5,
    smoothing=params.error_gen_smoothing,
    simple_error_rate=params.simple_error_rate,
    models_fpath=params.error_model
)

dset_kwargs = {
    'dset_fname': params.dset_path,
    'seq_length': params.seq_length,
    'padding_amt': params.padding_amt,
    'dataset_proportion': params.dataset_proportion,
    'vocabulary': v
}
dset_tr = dl.AgnosticOMRDataset(base='train', **dset_kwargs)
dset_vl = dl.AgnosticOMRDataset(base='validate', **dset_kwargs)
dset_tst = dl.AgnosticOMRDataset(base='test', **dset_kwargs)

dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
dloader_tst = DataLoader(dset_tst, params.batch_size, pin_memory=True)

model = lstm.LSTMModel(vocab_size=v.num_words, seq_length=params.seq_length).to(device)
model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
model = model.float()
model_size = sum(p.numel() for p in model.parameters())
logging.info(f'created model with n_params={model_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **params.scheduler_settings)

class_ratio = (1 / params.simple_error_rate)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(class_ratio))

logging.info('beginning training')
start_time = time.time()
val_losses = []
train_losses = []
best_model = None
tr_funcs.log_gpu_info()

if dry_run:
    assert False, "Dry run successful"

run_epoch_kwargs = {
    'model': model,
    'optimizer': optimizer,
    'criterion': criterion,
    'device': device,
    'example_generator': error_generator,
}


for epoch in range(params.num_epochs):
    epoch_start_time = time.time()

    # perform training epoch
    model.train()
    train_loss, tr_exs = tr_funcs.run_epoch(
        dloader=dloader,
        train=True,
        log_each_batch=True,
        **run_epoch_kwargs
    )

    # test on validation set
    model.eval()
    num_entries = 0
    val_loss = 0.
    with torch.no_grad():
        val_loss, val_exs = tr_funcs.run_epoch(
            dloader=dloader_val,
            train=False,
            log_each_batch=False,
            **run_epoch_kwargs
        )

    val_losses.append(val_loss)
    train_losses.append(train_loss)

    scheduler.step(val_loss)
    # tr_funcs.log_gpu_info()

    tr_f1, tr_thresh = ttnm.multilabel_thresholding(tr_exs['output'], tr_exs['target'])
    val_f1 = ttnm.f_measure(val_exs['output'].cpu(), val_exs['target'].cpu(), tr_thresh)

    epoch_end_time = time.time()
    logging.info(
        f'epoch {epoch:3d} | '
        f's/epoch    {(epoch_end_time - epoch_start_time):3.5e} | '
        f'train_loss {train_loss:1.6e} | '
        f'val_loss   {val_loss:1.6e} | '
        f'tr_thresh  {tr_thresh:1.5f} | '
        f'tr_f1      {tr_f1:1.6f} | '
        f'val_f1     {val_f1:1.6f} | '
    )

    # save an image
    if not epoch % params.save_img_every and epoch > 0:
        img_fpath = f'./out_imgs/epoch_{epoch}_{params.params_id_str}.txt'
        lines = po.plot_agnostic_results(tr_exs, v, tr_thresh)
        with open(img_fpath, 'w') as f:
            f.write(''.join(lines))

    # save a model checkpoint
    # if (not epoch % params.save_model_every) and epoch > 0 and params.save_model_every > 0:
    #     m_name = (
    #         f'lstm_{params.start_training_time}'
    #         f'_{params.lstm_summary_str}.pt')
    #     torch.save(cur_model, m_name)

    # keep snapshot of best model
    cur_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_losses': val_losses,
            }
    if len(val_losses) == 1 or val_losses[-1] < min(val_losses[:-1]):
        best_model = copy.deepcopy(cur_model)

    # early stopping
    time_since_best = epoch - train_losses.index(min(train_losses))
    if time_since_best > params.early_stopping_patience:
        logging.info(f'stopping early at epoch {epoch} because validation score stopped increasing')
        break

    elapsed = time.time() - start_time
    if elapsed > (params.max_time_minutes * 60):
        logging.info(f'stopping early at epoch {epoch} because of time limit')
        break

end_time = time.time()
logging.info(
    f'Training over at epoch at epoch {epoch}.\n'
    f'Total training time: {end_time - start_time} s.'
)

test_results = ttnm.TestResults(tr_thresh)
with torch.no_grad():
    tst_loss, tst_exs = tr_funcs.run_epoch(
        dloader=dloader_tst,
        train=False,
        log_each_batch=False,
        test_results=test_results,
        **run_epoch_kwargs
    )
res_stats = test_results.calculate_stats()
logging.info(
    f'precision: {res_stats["precision"]:1.6e} | '
    f'recall:    {res_stats["recall"]:1.6e} | '
    f'true positive: {res_stats["true positive rate"]:1.6e} | '
    f'true negative:   {res_stats["true negative rate"]:1.6e}'
)


for i in range(3):
    img_fpath = f'./out_imgs/FINAL_{i}_{params.params_id_str}.txt'
    lines = po.plot_agnostic_results(tst_exs, v, tr_thresh)
    with open(img_fpath, 'w') as f:
        f.write(''.join(lines))
