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
quartets_root = r"D:\Documents\datasets\just_quartets"

# keys = 
all_keys = ['ABC', 'kernscores', 'felix', 'felix_errors']
# keys = ['felix_errors']
c = m21.converter.Converter()

# parse all files and build vocabulary, first.
all_tokens = Counter()
print('building vocabulary...')
for k in all_keys:
    files = os.listdir(os.path.join(quartets_root, k))
    for fname in files:
        print(f'processing {fname}...')
        fpath = os.path.join(os.path.join(quartets_root, k, fname))
        parsed_file = m21.converter.parse(fpath)
        for p in list(parsed_file.getElementsByClass(m21.stream.Part)):
            agnostic = sta.m21_part_to_agnostic(p)
            all_tokens.update(agnostic) 
v = vocab.Vocabulary(all_tokens)
v.save_vocabulary('./data_management/vocab.txt')

# then parse them again to actually save them. yeah yeah this is not great
def make_hdf5(dset_path, keys, train_test_split=True):
    with h5py.File(dset_path, 'a') as f:
        f.attrs['beat_multiplier'] = beat_multiplier
        if train_val_test_split:
            train_grp = f.create_group('train')
            test_grp = f.create_group('test')
            validate_grp = f.create_group('validate')

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
            agnostics = [v.words_to_vec(sta.m21_part_to_agnostic(p)) for p in parts]
            arr = np.concatenate(agnostics)

            with h5py.File(dset_path, 'a') as f:
                if not train_val_test_split:
                    selected_subgrp = f
                elif i <= split_test:
                    selected_subgrp = f['test']
                elif i <= split_validate:
                    selected_subgrp = f['validate']
                else:
                    selected_subgrp = f['train']

                name = rf'{k}-{fname}'
                dset = selected_subgrp.create_dataset(
                    name=name,
                    data=arr,
                    compression='gzip'
                )

# make_hdf5(r'./all_string_quartets_agnostic.h5', ['ABC', 'kernscores', 'felix'], True)
make_hdf5(r'./quartets_felix_omr_agnostic.h5', ['felix_errors', 'felix'], False)