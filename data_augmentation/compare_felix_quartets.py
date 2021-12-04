import h5py
import numpy as np
import data_augmentation.needleman_wunsch_alignment as align
from numba import njit
from collections import Counter
import data_augmentation.error_gen_logistic_regression as elgr

dset_path = r'./processed_datasets/quartets_felix_omr_agnostic.h5'
ngram = 5 # n for n-grams for maxent markov model

with h5py.File(dset_path, 'r') as f:
    correct_fnames = [x for x in f.keys() if not 'omr' in x and not 'op80' in x]
    error_fnames = [x for x in f.keys() if 'omr' in x]
    correct_fnames = sorted(correct_fnames)
    error_fnames = sorted(error_fnames)
    correct_dset = [f[x][:].astype(np.uint8) for x in correct_fnames]
    error_dset = [f[x][:].astype(np.uint8) for x in error_fnames]

error_notes = {x:[] for x in ['replace_mod', 'insert_mod']}
correct_seqs_all = []
all_align_records = []

# training samples for logistic regression (MaxEnt Markov Model) for creating errors
# features in X are: [ngram of past 5 classes || note vector]
X = []
Y = []
for ind in range(len(correct_dset)):
        
    print(f'aligning {correct_fnames[ind]}...' )
    correct_seq = [x for x in correct_dset[ind]]
    error_seq = [x for x in error_dset[ind]]
    correct_align, error_align, r, score = align.perform_alignment(correct_seq, error_seq, match_weights=[3, -2], gap_penalties=[-2, -2, -1, -1])

    print(''.join(r))

    all_align_records.append(r)
    correct_seqs_all.extend(correct_seq)
    errors = []

    err_to_class = {'O': 0, '~': 1, '+': 2, '-': 3}
    most_recent_correct_note = 0

    for i in range(len(correct_align)):

        error_note = error_align[i]
        correct_note = correct_align[i]
        if r[i] == '~':
            # res = correct_note - error_note
            # error_notes['replace_mod'].append(res)
            error_notes['replace_mod'].append(error_note)
        elif r[i] == '+' and type(error_note) != str:
            error_notes['insert_mod'].append(error_note)

        if not type(correct_note) == str:
            most_recent_correct_note = correct_note
        prev_entries = [err_to_class[r[i - x]] if i - x >= 0  else err_to_class[r[0]] for x in range(1, ngram + 1)]
        
        # sample = np.concatenate([prev_entries, most_recent_correct_note])
        sample = np.array(prev_entries + [most_recent_correct_note])
        label = err_to_class[r[i]]
        X.append(sample)
        Y.append(label)

err_gen = elgr.ErrorGenerator(ngram, labeled_data=[X,Y], repl_samples=error_notes['replace_mod'], ins_samples=error_notes['insert_mod'])
err_gen.save_models('./data_augmentation/quartet_omr_error_models.joblib')
err_gen = elgr.ErrorGenerator(ngram, models_fpath='./data_augmentation/quartet_omr_error_models.joblib' )

with h5py.File(dset_path, 'r') as f:
    correct_fnames = [x for x in f.keys() if not 'omr' in x and x[-1] == '0' and not 'op80' in x]
    error_fnames = [x for x in f.keys() if 'omr' in x and x[-1] == '0']
    correct_fnames = sorted(correct_fnames)
    error_fnames = sorted(error_fnames)
    correct_dset = [f[x][:].astype(np.uint8) for x in correct_fnames]
    error_dset = [f[x][:].astype(np.uint8) for x in error_fnames]

# now, make .h5 file of test sequences
for ind in range(20, len(correct_dset)):
    print(f'aligning {error_fnames[ind]}...')

    correct_seq = correct_dset[ind]
    error_seq = error_dset[ind]

    err, Y = err_gen.add_errors_to_seq(correct_seq, error_seq)
    arr = np.stack([err, Y])    

    with h5py.File('./processed_datasets/supervised_omr_targets.h5', 'a') as f:
        name = error_fnames[ind]
        dset = f.create_dataset(
            name=name,
            data=arr,
            compression='gzip'
        )



