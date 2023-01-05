import torch
import torch.nn as nn
import agnostic_omr_dataloader as dl
import test_results_metrics as ttm
import models.LSTUT_model as lstut
import data_augmentation.error_gen_logistic_regression as err_gen
import training_helper_functions as tr_funcs
import models.LSTUT_model as lstut
import data_augmentation.error_gen_logistic_regression as err_gen
import training_helper_functions as tr_funcs
import data_management.vocabulary as vocab
import plot_outputs as po
import numpy as np
import wandb
from torch.utils.data import DataLoader
from collections import namedtuple
import model_params


# make dloaders for all test datasets identified in parameters file
def make_test_dataloaders(params, kwargs_dict):
    all_dset_groups = []

    EndGroup = namedtuple('TestGroup', 'dset dloader name with_targets')

    for test_set in params.test_sets:
        new_kwargs = dict(kwargs_dict)

        if test_set['with_targets']:
            new_kwargs['dset_fname'] = params.dset_testing_path
        else:
            new_kwargs['dset_fname'] = params.dset_path

        test_dset = dl.AgnosticOMRDataset(base=test_set['base'], **new_kwargs)
        dloader_omr = DataLoader(test_dset, params.batch_size, pin_memory=True)
        all_dset_groups.append(EndGroup(test_dset, dloader_omr, test_set['base'], test_set['with_targets']))
    return all_dset_groups

def add_stats_to_wandb(res_stats, target_recalls, end_name):
    for i, thresh in enumerate(target_recalls):
        wandb.run.summary[f"{end_name}_{thresh}_precision"] = res_stats["precision"][thresh]
        wandb.run.summary[f"{end_name}_{thresh}_true_negative"] = res_stats["true negative rate"][thresh]
        wandb.run.summary[f"{end_name}_{thresh}_prop_positive_predictions"] = res_stats["prop_positive_predictions"][thresh]
        wandb.run.summary[f"{end_name}_{thresh}_prop_positive_targets"] = res_stats["prop_positive_targets"][thresh]

def save_examples_to_wandb(res_stats, tst_exs, v, target_recalls, end_name, num_examples_to_save):
    wandb_dict = {}
    num_examples_to_save = min(num_examples_to_save, len(tst_exs['output']))

    for j, thresh in enumerate(res_stats['threshes']):
        
        # the 0 represents thresh optimized for f1 score instead
        target_recalls = target_recalls + ['F1']

        inds_to_save = np.random.choice(len(tst_exs['output']), num_examples_to_save, replace=False)
        for ind_to_save in (inds_to_save):
            
            batch_name = f"{tst_exs['batch_names'][ind_to_save]} {tst_exs['batch_offsets'][ind_to_save]}"
            lines = po.plot_agnostic_results(tst_exs, v, thresh, return_arrays=True, ind=ind_to_save)
            table = wandb.Table(data=lines, columns=['ORIG', 'INPUT', 'TARGET', 'OUTPUT', 'RAW'])
            wandb_dict[f'{end_name}_{target_recalls[j]}_{batch_name}'] = table
        
    wandb.run.summary[f'final_examples'] = wandb_dict

if __name__ == "__main__":
    model_path = "trained_models\lstut_best_LSTUT_TRIAL_0_(2022.12.28.17.22)_1-1-1-11-1-32-32.pt"

    params = model_params.Params('./param_sets/trial_lstut.json', False, 0)
    device, num_gpus = tr_funcs.get_cuda_info()

    v = vocab.Vocabulary(load_from_file=params.saved_vocabulary)
    error_generator = err_gen.ErrorGenerator(
        simple=params.simple_errors,
        smoothing=params.error_gen_smoothing,
        simple_error_rate=params.simple_error_rate,
        parallel=params.errors_parallel,
        models_fpath=params.error_model
    )

    lstut_settings = params.lstut_settings
    lstut_settings['vocab_size'] = v.num_words
    lstut_settings['seq_length'] = params.seq_length
    model = lstut.LSTUT(**lstut_settings).to(device)
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model = model.float()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    model_size = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={model_size}')

    dset_kwargs = {
        'dset_fname': params.dset_path,
        'seq_length': params.seq_length,
        'padding_amt': params.padding_amt,
        'dataset_proportion': 1, #params.dataset_proportion,
        'vocabulary': v
    }

    run_epoch_kwargs = {
        'model': model,
        'optimizer': None,
        'criterion': criterion,
        'device': device,
        'example_generator': error_generator,
    }

    groups =  make_test_dataloaders(params, dset_kwargs)

    for g in groups:
        res_stats, tst_exs, test_results = tr_funcs.test_end_group(
            g.dloader,
            g.with_targets,
            run_epoch_kwargs,
            params.target_recalls
            )