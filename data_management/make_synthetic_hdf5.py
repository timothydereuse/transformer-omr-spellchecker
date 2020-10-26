import h5py
import numpy as np
import model_params as params

f = h5py.File('synthetic_repetition_dset.hdf5', 'a')
num_sequences = 20000
val_range = 50
len_range = (5, 15)
repeat_range = (3, 6)

train_grp = f.create_group('train')
test_grp = f.create_group('test')
validate_grp = f.create_group('validate')

# iterate over all datasets
train_subgrp = train_grp.create_group('syn')
test_subgrp = test_grp.create_group('syn')
validate_subgrp = validate_grp.create_group('syn')

split_test = np.round(params.test_proportion * num_sequences)
split_validate = np.round(params.validate_proportion * num_sequences) + split_test

# iterate over all krn files in the current dataset
for i in range(num_sequences):

    if not i % 100:
        print(f'creating sequence {i} of {num_sequences}...')

    subs_len = np.random.randint(len_range[0], len_range[1])
    subs_rpt = np.random.randint(repeat_range[0], repeat_range[1])
    subs = np.random.randint(1, val_range, subs_len)
    arr = np.concatenate([subs for _ in range(subs_rpt)])

    if i <= split_test:
        selected_subgrp = test_subgrp
    elif i <= split_validate:
        selected_subgrp = validate_subgrp
    else:
        selected_subgrp = train_subgrp

    dset = selected_subgrp.create_dataset(
        name=f'syn_{i}',
        data=arr
    )

f.close()
