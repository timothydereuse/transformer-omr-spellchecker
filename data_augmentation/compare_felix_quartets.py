import h5py
import numpy as np
import data_augmentation.needleman_wunsch_alignment as align
from numba import njit
from collections import Counter
import data_augmentation.error_gen_logistic_regression as elgr

dset_path = r'./processed_datasets/quartets_felix_omr_agnostic2.h5'

with h5py.File(dset_path, 'r') as f:
    correct_fnames = sorted([x for x in f.keys() if 'correct' in x])
    error_fnames = sorted([x for x in f.keys() if 'omr' in x])
    error_onepass_fnames = sorted([x for x in f.keys() if 'onepass' in x])

    correct_dset = [f[x][:].astype(np.uint8) for x in correct_fnames]
    error_dset = [f[x][:].astype(np.uint8) for x in error_fnames]
    error_onepass_dset = [f[x][:].astype(np.uint8) for x in error_onepass_fnames]

error_notes = {x:[] for x in ['replace_mod', 'insert_mod']}
correct_seqs_all = []
all_align_records = []

def get_training_samples(correct_dset, error_dset):
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

            # this horrible string hack is necessary beause sklearn's label encoder takes only strings or numbers as input,
            # not tuples. everything is a string of '[error type].[replace/insertion entry]'
            c = err_to_class[r[i]]
            if r[i] == 'O':
                label = f'{c}.0'
            elif r[i] == '~':
                error_notes['replace_mod'].append(error_note)
                label = f'{c}.{error_note}'
            elif r[i] == '+':
                error_notes['insert_mod'].append(error_note)
                label = f'{c}.{error_note}'
            elif r[i] == '-':
                label = f'{c}.0'

            if not type(correct_note) == str:
                most_recent_correct_note = correct_note
            # prev_entries = [err_to_class[r[i - x]] if i - x >= 0  else err_to_class[r[0]] for x in range(1, ngram + 1)]
            sample = most_recent_correct_note 

            X.append(sample)
            Y.append(label)
    return X, Y

X, Y = get_training_samples(correct_dset, error_dset)
X = np.array(X).reshape(-1, 1)
err_gen = elgr.ErrorGenerator(labeled_data=[X,Y])
err_gen.save_models('./data_augmentation/quartet_omr_error_models.joblib')
# err_gen = elgr.ErrorGenerator(3, models_fpath='./data_augmentation/quartet_omr_error_models.joblib' )

# now, make .h5 file of test sequences

pms = [
    (error_dset, error_fnames, 'omr'), 
    (error_onepass_dset, error_onepass_fnames, 'onepass')
]

# X, Y = get_training_samples(correct_dset, error_onepass_dset)

for t in pms:
    target_dset, target_dset_fnames, category = t

    with h5py.File('./processed_datasets/supervised_omr_targets2.h5', 'a') as f:
        f.create_group(category)

    # X, Y = get_training_samples(correct_dset, target_dset)
    for ind in range(len(correct_dset)):
        print(f'aligning {error_fnames[ind]}...')

        correct_seq = correct_dset[ind]
        error_seq = target_dset[ind]

        err, Y = err_gen.add_errors_to_seq(correct_seq, error_seq)
        arr = np.stack([err, Y])    

        with h5py.File('./processed_datasets/supervised_omr_targets2.h5', 'a') as f:
            name = target_dset_fnames[ind]
            g = f[category]
            dset = g.create_dataset(
                name=name,
                data=arr,
                compression='gzip'
            )


# sharp_probs = e.regression.predict_proba(e.enc.transform([[198]]))[0] 
# [
#     (
#         x, 
#         sharp_probs[x],
#         e.enc_labels.inverse_transform([x])[0][0],
#         v.vec_to_words([int(e.enc_labels.inverse_transform([x])[0].split('.')[-1])])
#     ) 
#         for x in np.argsort(sharp_probs)
# ]

