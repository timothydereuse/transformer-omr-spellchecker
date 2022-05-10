import music21 as m21
import os
from collections import namedtuple, Counter
import numpy as np
import h5py
import data_management.semantic_to_agnostic as sta
import data_management.vocabulary as vocab

test_proportion = 0.1
validate_proportion = 0.1
beat_multiplier = 48
num_transpositions_per_file = 3
possible_transpositions = ['m2', 'M2', 'm3', 'M3', 'P4', 'a4', 'd5', 'p5']
possible_transpositions = possible_transpositions + ['-' + x for x in possible_transpositions]
quartets_root = r"/Users/tim/Documents/Datasets/felix_quartets"

# all_keys = ['ABC', 'kernscores', 'felix', 'felix_errors']
c = m21.converter.Converter()

# # parse all files and build vocabulary, first.
# all_tokens = Counter()
# print('building vocabulary...')
# for k in all_keys:
#     files = os.listdir(os.path.join(quartets_root, k))
#     for fname in files:
#         print(f'processing {fname}...')
#         fpath = os.path.join(os.path.join(quartets_root, k, fname))
#         parsed_file = m21.converter.parse(fpath)
#         parts = list(parsed_file.getElementsByClass(m21.stream.Part))
#         agnostic = sta.m21_parts_to_interleaved_agnostic(parts, remove=['+'], just_tokens=True)
#         all_tokens.update(agnostic) 
# v = vocab.Vocabulary(all_tokens)
# v.save_vocabulary('./data_management/vocab.txt')
v = vocab.Vocabulary(load_from_file='./data_management/vocab.txt')

# then parse them again to actually save them. yeah, yeah, this is not great
def make_hdf5(dset_path, keys, train_val_test_split=True, split_by_keys=False, transpose=False):
    with h5py.File(dset_path, 'a') as f:
        f.attrs['beat_multiplier'] = beat_multiplier

        if train_val_test_split:
            train_grp = f.create_group('train')
            test_grp = f.create_group('test')
            validate_grp = f.create_group('validate')
        elif split_by_keys:
            grps = {k:f.create_group(k) for k in keys}

    for k in keys:
        files = os.listdir(os.path.join(quartets_root, k))
        np.random.shuffle(files)

        split_test = int(np.round(test_proportion * len(files)))
        split_validate = int(np.round(validate_proportion * len(files)) + split_test)

        for i, fname in enumerate(files):
            print(f'parsing {k}/{fname}...')

            fpath = os.path.join(quartets_root, k, fname)
            try:
                parsed_file = m21.converter.parse(fpath)
            except Exception:
                print(f'parsing {k}/{fname} failed, skipping file')
                continue
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))

            if transpose:
                transpositions = np.random.choice(possible_transpositions, num_transpositions_per_file, replace=False)
                transpositions = np.concatenate([[None], transpositions])
            else:
                transpositions = [None]

            agnostics = [
                sta.m21_parts_to_interleaved_agnostic(parts, transpose=x, remove=['+'], just_tokens=True) 
                for x in transpositions
            ]
            agnostic_vecs = [v.words_to_vec(x) for x in agnostics]
            arrs = [np.array(x) for x in agnostic_vecs]

            with h5py.File(dset_path, 'a') as f:
                if not train_val_test_split:
                    selected_subgrp = f
                elif i <= split_test:
                    selected_subgrp = f['test']
                elif i <= split_validate:
                    selected_subgrp = f['validate']
                else:
                    selected_subgrp = f['train']

                if split_by_keys:
                    selected_subgrp = f[k]

                for i, arr in enumerate(arrs):
                    name = rf'{k}-{fname}-{i}'
                    selected_subgrp.create_dataset(
                        name=name,
                        data=arr,
                        compression='gzip'
                    )

# make_hdf5(r'./processed_datasets/all_string_quartets_agnostic.h5', ['ABC', 'kernscores', 'felix'], True)
make_hdf5(r'./processed_datasets/quartets_felix_omr_agnostic2.h5', ['felix_omr', 'felix_correct', 'felix_onepass'], False)
