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

################################################
# SETTING UP DATASETS AND MODEL FOR TRAINING
################################################

parser = argparse.ArgumentParser(description=
    'Training and testing script for the transformer-omr-spellchecker project. '
    'Must reference a .json parameters file (in the /param_sets folder. '
    'Requires processed .h5 files containing quartets in agnostic format; '
    'some of these .h5 files are included with the transformer-omr-spellchecker repository on GitHub. '
    'Use the script run_all_data_preparation to make these files from scratch, or from another dataset.')
parser.add_argument('parameters', default='default_params.json',
                    help='Parameter file in .json format.')
parser.add_argument('-m', '--mod_number', type=int, default=0,
                    help='Index of specific modification to apply to given parameter set.')
parser.add_argument('-w', '--wandb', type=ascii, action='store', default=None,
                    help='Name of wandb project to log results to. '
                         'Results are otherwise just printed to stdout and log file if -l is used.')
parser.add_argument('-l', '--logging', action='store_true',
                    help='Whether or not to log training results to file.')
parser.add_argument('-d', '--dryrun', action='store_true',
                    help='Halts execution immediately before training begins.')
args = vars(parser.parse_args())

params = model_params.Params(args['parameters'],  args['logging'], args['mod_number'])
dry_run = args['dryrun']
run_name = params.params_id_str + ' ' + params.mod_string

if (not dry_run) and args['wandb']:
    wandb.init(project=args['wandb'].strip("'"), config=params.params_dict, entity="timothydereuse")
    wandb.run.name = run_name

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
dset_kwargs['dataset_proportion'] = 1
dset_kwargs['dset_fname'] = params.dset_testing_path
dset_omr = dl.AgnosticOMRDataset(base='omr', **dset_kwargs)
dset_omr_onepass = dl.AgnosticOMRDataset(base='onepass', **dset_kwargs)

dloader = DataLoader(dset_tr, params.batch_size, pin_memory=True)
dloader_val = DataLoader(dset_vl, params.batch_size, pin_memory=True)
dloader_tst = DataLoader(dset_tst, params.batch_size, pin_memory=True)
dloader_omr = DataLoader(dset_omr, params.batch_size, pin_memory=True)
dloader_omr_onepass = DataLoader(dset_omr_onepass, params.batch_size, pin_memory=True)

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

class_ratio = (params.error_gen_smoothing)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(class_ratio))

#########################
# TRAINING MODEL
#########################

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
        log_each_batch=False,
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

    # get thresholds that maximize f1 and match required recall scores
    sig_val_output = torch.sigmoid(val_exs['output'])
    sig_train_output = torch.sigmoid(tr_exs['output'])
    tr_f1, tr_thresh = ttm.multilabel_thresholding(sig_train_output, tr_exs['target'], beta=class_ratio)
    val_f1 = ttm.f_measure(sig_val_output.cpu(), val_exs['target'].cpu(), tr_thresh, beta=class_ratio)
    # val_threshes = ttm.find_thresh_for_given_recalls(sig_val_output.cpu(), val_exs['target'].cpu(), params.target_recalls)

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

    if args['wandb']:
        wandb.log({
            'epoch_s': (epoch_end_time - epoch_start_time), 
            'train_loss': train_loss,
            'val_loss': val_loss,
            'tr_thresh': tr_thresh,
            'tr_f1': tr_f1,
            'val_f1': val_f1
            })

    # save an example
    if args['wandb'] and (not epoch % params.save_img_every) and epoch > 0:
        lines = po.plot_agnostic_results(tr_exs, v, tr_thresh, return_arrays=True)
        table =  wandb.Table(data=lines, columns=['ORIG', 'INPUT', 'TARGET', 'OUTPUT', 'RAW'])
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
    time_since_best = epoch - val_losses.index(min(val_losses))
    if time_since_best > params.early_stopping_patience:
        print(f'stopping early at epoch {epoch} because validation score stopped increasing')
        break

    # stopping based on time limit defined in params file
    elapsed = time.time() - start_time
    if elapsed > (params.max_time_minutes * 60):
        print(f'stopping early at epoch {epoch} because of time limit')
        break

end_time = time.time()
print(
    f'Training over at epoch at epoch {epoch}.\n'
    f'Total training time: {end_time - start_time} s.'
)

#########################
# TESTING TRAINED MODEL 
#########################

# define (dloader, group name, end_train_data_mode) for each end group to test through
end_groups = [
    (dloader_tst, 'data_aug_test', False),
    (dloader_omr, 'real_omr_test', True),
    (dloader_omr_onepass, 'real_onepass_test', True)
    ]

for end_group in end_groups:
    end_dloader, end_name, end_train_data_mode = end_group

    res_stats, tst_exs, test_results = tr_funcs.test_end_group(end_group, run_epoch_kwargs, params.target_recalls)

    print(
        f'{end_name} STATS:\n'
        f'{end_name} threshes:                    {res_stats["threshes"]}\n'
        f'{end_name}_precision:                   {res_stats["precision"]} | \n'
        f'{end_name}_recall:                      {res_stats["recall"]} | \n'
        f'{end_name}_true negative:               {res_stats["true negative rate"]} \n'
        f'{end_name}_prop_positive_predictions:   {res_stats["prop_positive_predictions"]} \n'
        f'{end_name}_prop_positive_targets:       {res_stats["prop_positive_targets"]} \n'
        f'{end_name}_max_f1:                      {res_stats["max_f1"]} \n'
    )

    if args['wandb']:
        wandb.run.summary["total_training_time"] = end_time - start_time
        for i, thresh in enumerate(params.target_recalls):
            # target_recall = params.target_recalls[i]
            wandb.run.summary[f"{end_name}_{thresh}_precision"] = res_stats["precision"][thresh]
            wandb.run.summary[f"{end_name}_{thresh}_true_negative"] = res_stats["true negative rate"][thresh]
            wandb.run.summary[f"{end_name}_{thresh}_prop_positive_predictions"] = res_stats["prop_positive_predictions"][thresh]
            wandb.run.summary[f"{end_name}_{thresh}_prop_positive_targets"] = res_stats["prop_positive_targets"][thresh]

    num_examples_to_save = min(params.num_examples_to_save, len(tst_exs['output']))
    for j, thresh in enumerate(res_stats['threshes']):
        wandb_dict = {}

        # all_predictions = (torch.sigmoid(tst_exs['output']) > thresh).float().mean()
        # print(all_predictions)

        # the 0 represents thresh optimized for f1 score instead
        target_recalls = params.target_recalls + ['F1']

        inds_to_save = np.random.choice(len(tst_exs['output']), num_examples_to_save, replace=False)
        for ind_to_save in (inds_to_save):
            
            batch_name = f"{tst_exs['batch_names'][ind_to_save]} {tst_exs['batch_offsets'][ind_to_save]}"
            lines = po.plot_agnostic_results(tst_exs, v, thresh, return_arrays=True, ind=ind_to_save)
            table = wandb.Table(data=lines, columns=['ORIG', 'INPUT', 'TARGET', 'OUTPUT', 'RAW'])

            wandb_dict[f'{end_name}_{inds_to_save}_{target_recalls[j]}_{batch_name}'] = table
        
        if args['wandb']:
            wandb.run.summary[f'final_examples'] = wandb_dict

# if max_epochs reached, or early stopping condition reached, save best model
best_epoch = best_model['epoch']
m_name = (f'lstut_best_{params.params_id_str}.pt')
torch.save(best_model, m_name)
