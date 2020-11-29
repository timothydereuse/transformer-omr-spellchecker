import h5py
import numpy as np
import music21 as m21
import os
import model_params as params

paths = params.raw_data_paths
f = h5py.File(params.dset_path, 'a')
f.attrs['beat_multiplier'] = params.beat_multiplier

train_grp = f.create_group('train')
test_grp = f.create_group('test')
validate_grp = f.create_group('validate')

# iterate over all datasets
for p in paths.keys():

    train_subgrp = train_grp.create_group(p)
    test_subgrp = test_grp.create_group(p)
    validate_subgrp = validate_grp.create_group(p)

    all_krns = []
    for root, dirs, files in os.walk(paths[p]):
        for name in files:
            if '.krn' in name:
                all_krns.append(os.path.join(root, name))
    np.random.shuffle(all_krns)

    split_test = np.round(params.test_proportion * len(all_krns))
    split_validate = np.round(params.validate_proportion * len(all_krns)) + split_test

    # iterate over all krn files in the current dataset
    for i, krn_fname in enumerate(all_krns):

        print(f'processing {krn_fname} | {i} of {len(all_krns)} in {p}')
        krn = m21.converter.parse(krn_fname)
        arr = np.array([[
            n.pitch.midi if n.isNote else 0,
            int(n.offset * params.beat_multiplier),
            int(n.duration.quarterLength * params.beat_multiplier)
        ] for n in krn.flat.notesAndRests])

        if i <= split_test:
            selected_subgrp = test_subgrp
        elif i <= split_validate:
            selected_subgrp = validate_subgrp
        else:
            selected_subgrp = train_subgrp

        dset = selected_subgrp.create_dataset(
            name=krn_fname.split('\\')[-1],
            data=arr
        )

        try:
            time_sig = list(krn.flat.getElementsByClass('TimeSignature'))[0]
            dset.attrs['time_signature'] = (time_sig.numerator, time_sig.denominator)
        except IndexError:
            dset.attrs['time_signature'] = (-1, -1)

        try:
            key = list(krn.flat.getElementsByClass('Key'))[0]
            dset.attrs['key'] = key.tonicPitchNameWithCase
        except IndexError:
            dset.attrs['key'] = -1
