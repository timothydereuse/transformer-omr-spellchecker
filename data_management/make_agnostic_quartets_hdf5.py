import music21 as m21
import os
from collections import namedtuple
import numpy as np
import h5py

test_proportion = 0.1
validate_proportion = 0.1
dset_path = r'./all_string_quartets_agnostic.h5'
beat_multiplier = 48
quartets_root = r"D:\Documents\datasets\just_quartets"
train_val_test_split = True

keys = ['ABC', 'kernscores', 'felix']
c = m21.converter.Converter()

with h5py.File(dset_path, 'a') as f:
    f.attrs['beat_multiplier'] = beat_multiplier
    if train_val_test_split:
        train_grp = f.create_group('train')
        test_grp = f.create_group('test')
        validate_grp = f.create_group('validate')

# parse all files and build vocabulary, first.
for k in keys:
    files = os.listdir(os.path.join(quartets_root, k))
    for fname in files:
        print(f'processing {fname}')
        fpath = os.path.join(os.path.join(quartets_root, k, fname))
        parsed_file = m21.converter.parse(fpath)
        for p in list(parsed_file.getElementsByClass(m21.stream.Part)):
            agnostic = sta.m21_part_to_agnostic(p)
            all_tokens.update(agnostic) 
v = Vocabulary(all_tokens)

# then parse them again to actually save them. yeah yeah this is not great
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
