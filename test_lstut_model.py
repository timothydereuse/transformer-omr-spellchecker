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
import data_management.semantic_to_agnostic as sta
import data_management.vocabulary as vocab
import plot_outputs as po
import numpy as np
import wandb
import music21 as m21
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


def assign_color_to_stream(this_stream, agnostic_rec, predictions, color_style='red'):
    # given a music21 stream, a list of agnostic tokens, a list of predictions
    # on those tokens, and a color, assign that color to all objects in the
    # music21 stream where the prediction on its associated token is true.

    parts = list(this_stream.getElementsByClass(m21.stream.Part))
    all_measures = [list(x.getElementsByClass(m21.stream.Measure)) for x in parts]
    num_tokens = len(agnostic_rec)

    # for every note predicted incorrect, find the element of the m21 stream
    # that corresponds to it
    for i, record, prediction in zip(range(num_tokens), agnostic_rec, predictions):

        # change nothing if this token is predicted correct
        if not prediction:
            continue

        # get the single m21 element referred to by this token
        selected_element = all_measures[record.part_idx][record.measure_idx][record.event_idx]
        token_type = record.agnostic_item.split('.')[0]

        # how do we handle chords? if it's a chord, then record.chord_idx will be non-zero:
        # choose the note in that chord corresponding to the relevant chord idx
        if type(selected_element) == m21.chord.Chord:
            selected_element = selected_element.notes[record.chord_idx]

        # now for a bunch of if statements to handle each individual type of token
        # in its own way, because i don't know what else to do
        if type(selected_element) == m21.layout.SystemLayout:
            # if the token of interest refers to a layout element in m21, then that
            # token is part of a time signature, a courtesy clef, or the restatement
            # of a key signature. None of these things can be easily marked individually
            # in MusicXML or Music21, so we ignore them.
            continue
        elif token_type == 'accid' and type(selected_element) == m21.note.Note:
            selected_element.pitch.accidental.style.color = color_style
        elif token_type == 'articulation' and type(selected_element) == m21.note.Note:
            for articulation in selected_element.articulations:
                articulation.style.color = color_style
        elif token_type == 'rest' and type(selected_element) == m21.note.Rest:
            selected_element.style.color = color_style
        elif token_type == 'note' and type(selected_element) == m21.note.Note:
            selected_element.style.color = color_style
        else: 
            selected_element.style.color = color_style
    
    return this_stream


def run_m21_streams_through_model(parsed_files, model, seq_length, vocab):
    model.eval()

    agnostic_records = sta.m21_streams_to_agnostic(parsed_files)
    output_agnostic_recs = []
    output_preds = []
    for agnostic_rec in agnostic_records:

        agnostic_tokens = [x.agnostic_item for x in agnostic_rec]
        vectorized = vocab.words_to_vec(agnostic_tokens).astype('long')
        inp = torch.tensor(vectorized)

        # expand input vector to be a multiple of the sequence length
        # and then reshape into correct form for prediction
        expand_amt = seq_length - (len(inp) % seq_length)
        expand_vec = torch.full((expand_amt,), vocab.SEQ_PAD)
        inp = torch.cat([inp, expand_vec])
        inp = inp.reshape(-1, seq_length)

        with torch.no_grad():
            pred = model(inp)

        # unwrap prediction
        unwrapped_pred = pred.reshape(-1)[:-expand_amt]

        output_agnostic_recs.append(agnostic_rec)
        output_preds.append(unwrapped_pred)
    
    return output_agnostic_recs, output_preds


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

    # for g in groups:
    #     res_stats, tst_exs, test_results = tr_funcs.test_end_group(
    #         g.dloader,
    #         g.with_targets,
    #         run_epoch_kwargs,
    #         params.target_recalls
    #         )

    test_files = [r"C:\Users\tim\Documents\felix_quartets_got_annotated\1_op12\C3\1_op12_2_aligned.musicxml"]
    parsed_files = [m21.converter.parse(fpath) for fpath in test_files]

    agnostic_records, predictions = run_m21_streams_through_model(parsed_files, model, params.seq_length, v)

    color_style = 'red'

    output_streams = []
    for i in range(len(parsed_files)):
        agnostic_rec = agnostic_records[i]
        this_stream = parsed_files[i]
        thresh_pred = (predictions[i] > torch.mean(predictions[i])).numpy()
        colored_stream = assign_color_to_stream(this_stream, agnostic_rec, thresh_pred, color_style)
        output_streams.append(colored_stream)

    output_streams[0].write('musicxml', fp='./test.musicxml')