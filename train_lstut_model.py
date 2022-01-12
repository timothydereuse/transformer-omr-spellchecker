import time, logging, argparse, copy
import numpy as np
import torch, wandb
import torch.nn as nn
import agnostic_omr_dataloader as dl
import test_trained_model as ttm
import models.LSTUT_model as lstut
import data_augmentation.error_gen_logistic_regression as err_gen
import training_helper_functions as tr_funcs
import data_management.vocabulary as vocab
from torch.utils.data import DataLoader
import plot_outputs as po
import model_params

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib import reload
reload(tr_funcs)
reload(dl)
reload(lstut)
reload(model_params)
reload(po)
reload(err_gen)
reload(ttm)

parser = argparse.ArgumentParser(description='Training script, with optional parameter searching.')
parser.add_argument('parameters', default='default_params.json',
                    help='Parameter file in .json format.')
parser.add_argument('-m', '--mod_number', type=int, default=0,
                    help='Index of specific modification to apply to given parameter set.')
parser.add_argument('-w', '--wandb', type=ascii, action='store', default=None,
                    help='Index of specific modification to apply to given parameter set.')
parser.add_argument('-l', '--logging', action='store_true',
                    help='Whether or not to log training results to file.')
parser.add_argument('-d', '--dryrun', action='store_true',
                    help='Halts execution immediately before training begins.')
args = vars(parser.parse_args())

params = model_params.Params(args['parameters'],  args['logging'], args['mod_number'])
dry_run = args['dryrun']

if (not dry_run) and args['wandb']:
    wandb.init(project=args['wandb'].strip("'"), config=params.params_dict, entity="timothydereuse")

device, num_gpus = tr_funcs.get_cuda_info()
print('defining datasets...')

v = vocab.Vocabulary(load_from_file=params.saved_vocabulary)
error_generator = err_gen.ErrorGenerator(
    simple=params.simple_errors,
    smoothing=params.error_gen_smoothing,
    simple_error_rate=params.simple_error_rate,
    parallel=params.errors_parallel,
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
dset_kwargs['dset_fname'] = params.dset_testing_path
dset_kwargs['dataset_proportion'] = 1
dset_omr = dl.AgnosticOMRDataset(base=None, **dset_kwargs)


dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
dloader_tst = DataLoader(dset_tst, params.batch_size, pin_memory=True)
dloader_omr = DataLoader(dset_omr, params.batch_size, pin_memory=True)

lstut_settings = params.lstut_settings
lstut_settings['vocab_size'] = v.num_words
lstut_settings['seq_length'] = params.seq_length
model = lstut.LSTUT(**lstut_settings).to(device)
model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
model = model.float()
model_size = sum(p.numel() for p in model.parameters())
print(f'created model with n_params={model_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **params.scheduler_settings)

class_ratio = (1 / params.simple_error_rate)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(class_ratio))

print('beginning training')
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

    tr_f1, tr_thresh = ttm.multilabel_thresholding(tr_exs['output'], tr_exs['target'])
    val_f1 = ttm.f_measure(val_exs['output'].cpu(), val_exs['target'].cpu(), tr_thresh)

    epoch_end_time = time.time()
    print(
        f'epoch {epoch:3d} | '
        f's/epoch    {(epoch_end_time - epoch_start_time):3.5e} | '
        f'train_loss {train_loss:1.6e} | '
        f'val_loss   {val_loss:1.6e} | '
        f'tr_thresh  {tr_thresh:1.5f} | '
        f'tr_f1      {tr_f1:1.6f} | '
        f'val_f1     {val_f1:1.6f} | '
    )

    if logging:
        wandb.log({
            'epoch_s': (epoch_end_time - epoch_start_time), 
            'train_loss': train_loss,
            'val_loss': val_loss,
            'tr_thresh': tr_thresh,
            'tr_f1': tr_f1,
            'val_f1': val_f1
            })

    # save an image
    if not epoch % params.save_img_every and epoch > 0:
        lines = po.plot_agnostic_results(tr_exs, v, tr_thresh, return_arrays=True)
        table =  wandb.Table(data=lines, columns=['ORIG', 'INPUT', 'TARGET', 'OUTPUT'])
        wandb.log({'examples': table})

    # save a model checkpoint
    # if (not epoch % params.save_model_every) and epoch > 0 and params.save_model_every > 0:
    #     m_name = (
    #         f'lstut_{params.start_training_time}'
    #         f'_{params.lstut_summary_str}.pt')
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
wandb.run.summary["total_training_time"] = end_time - start_time
print(
    f'Training over at epoch at epoch {epoch}.\n'
    f'Total training time: {end_time - start_time} s.'
)

for end_group in [(dloader_tst, 'data_aug_test'), (dloader_omr, 'real_omr_test')]:
    end_dloader, end_name = end_group
    test_results = ttm.TestResults(tr_thresh)
    with torch.no_grad():
        tst_loss, tst_exs = tr_funcs.run_epoch(
            dloader=dloader_tst,
            train=False,
            log_each_batch=False,
            test_results=test_results,
            **run_epoch_kwargs
        )
    res_stats = test_results.calculate_stats()
    print(
        f'{end_name}_precision: {res_stats["precision"]:1.6e} | '
        f'{end_name}_recall:    {res_stats["recall"]:1.6e} | '
        f'{end_name}_true positive: {res_stats["true positive rate"]:1.6e} | '
        f'{end_name}_true negative:   {res_stats["true negative rate"]:1.6e}'
    )

    wandb.run.summary[f"{end_name}_precision"] = res_stats["precision"]
    wandb.run.summary[f"{end_name}_recall"] = res_stats["recall"]
    wandb.run.summary[f"{end_name}_true_positive"] = res_stats["true positive rate"]
    wandb.run.summary[f"{end_name}_true_negative"] = res_stats["true negative rate"]

    for i in range(3):
        lines = po.plot_agnostic_results(tr_exs, v, tr_thresh, return_arrays=True)
        table = wandb.Table(data=lines, columns=['ORIG', 'INPUT', 'TARGET', 'OUTPUT'])
        wandb.run.summary[f'{end_name}_examples_final'] = table

# # if max_epochs reached, or early stopping condition reached, save best model
# best_epoch = best_model['epoch']
# m_name = (f'lstut_best_{params.start_training_time}_{params.lstut_summary_str}.pt')
# torch.save(best_model, m_name)
