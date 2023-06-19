from data_augmentation.compare_paired_quartets import (
    get_training_samples,
    make_supervised_examples,
)
from data_augmentation.error_gen_logistic_regression import ErrorGenerator
from data_management.make_agnostic_quartets_hdf5 import make_hdf5, build_vocab
from data_management.vocabulary import Vocabulary
import h5py
import numpy as np

quartets_root = r"C:\Users\tim\Documents\datasets\just_quartets"
paired_quartets_root = (
    r"C:\Users\tim\Documents\datasets\just_quartets\paired_omr_correct"
)

all_quartets_dset_path = r"./processed_datasets/all_quartets_bymeasure.h5"
paired_dset_path = r"./processed_datasets/paired_quartets_bymeasure.h5"
supervised_targets_fname = r"./processed_datasets/supervised_omr_targets_bymeasure.h5"
error_generator_fname = (
    r"./processed_datasets/paired_quartets_error_model_bymeasure.joblib"
)
vocab_name = r"./processed_datasets/vocab_big.txt"
interleave = True

build_vocab(
    all_keys=[
        "paired_omr_correct/correct_quartets",
        "paired_omr_correct/omr_quartets",
        "paired_omr_correct/onepass_quartets",
        "musescore_misc",
        "jsb_choral_krn",
        "jsb_fakes_mxl",
        "musescore_misc_pd",
        "ABC",
        "kernscores",
        "musescore_misc",
    ],
    out_fname=vocab_name,
    quartets_root=quartets_root,
)

v = Vocabulary(load_from_file=vocab_name)

# make hdf5 of CORRECT string quartets
make_hdf5(
    all_quartets_dset_path,
    [
        "jsb_fakes_mxl",
        "jsb_choral_krn",
        "musescore_misc",
        "musescore_misc_pd",
        "ABC",
        "kernscores",
    ],
    v=v,
    quartets_root=quartets_root,
    train_val_test_split=True,
    split_by_keys=False,
    transpose=True,
    interleave=interleave,
)

# make intermediate hdf5 of CORRECT / ERROR PAIRED string quartets to later do sequence alignment on
# (this hdf5 is not actually used in the training process)
make_hdf5(
    paired_dset_path,
    ["paired_correct", "paired_omr", "paired_onepass"],
    v=v,
    quartets_root=paired_quartets_root,
    train_val_test_split=False,
    split_by_keys=True,
    transpose=False,
    interleave=interleave,
)

with h5py.File(paired_dset_path, "r") as f:
    correct_fnames = sorted(list(f["paired_correct"].keys()))
    error_fnames = sorted(list(f["paired_omr"].keys()))
    error_onepass_fnames = sorted(list(f["paired_onepass"].keys()))

    correct_dset = [f["paired_correct"][x][:].astype(np.uint16) for x in correct_fnames]
    error_dset = [f["paired_omr"][x][:].astype(np.uint16) for x in error_fnames]
    error_onepass_dset = [
        f["paired_onepass"][x][:].astype(np.uint16) for x in error_onepass_fnames
    ]


X, Y = get_training_samples(correct_dset, error_dset, correct_fnames)
X = np.array(X).reshape(-1, 1)
err_gen = ErrorGenerator(labeled_data=[X, Y])
err_gen.save_models(error_generator_fname)

# now, make .h5 file of test sequences
pms = [
    (error_dset, error_fnames, "omr"),
    (error_onepass_dset, error_onepass_fnames, "onepass"),
]

make_supervised_examples(pms, supervised_targets_fname, err_gen, correct_dset)
