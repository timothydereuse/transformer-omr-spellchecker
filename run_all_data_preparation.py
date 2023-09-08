from data_augmentation.compare_paired_quartets import (
    get_training_samples,
    make_supervised_examples,
)
from data_augmentation.error_gen_logistic_regression import ErrorGenerator
from data_management.make_agnostic_quartets_hdf5 import make_hdf5, build_vocab
from data_management.vocabulary import Vocabulary
import h5py
import numpy as np
import os

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

# ensure that omr vs correct filenames are identical
correct_quart_path = os.path.join(paired_quartets_root, "correct_quartets")
omr_quart_path = os.path.join(paired_quartets_root, "omr_quartets")
cor = set(["".join(x.split(".")[:-1]) for x in os.listdir(correct_quart_path)])
err = set(["".join(x.split(".")[:-1]) for x in os.listdir(omr_quart_path)])
assert not (cor.difference(err)), "filenames not synced up cor -> err"
assert not (err.difference(cor)), "filenames not synced up err -> cor"
print("filenames checked - correct")

# make intermediate hdf5 of CORRECT / ERROR PAIRED string quartets to later do sequence alignment on
# (this hdf5 is not actually used in the training process)
make_hdf5(
    paired_dset_path,
    ["correct_quartets", "omr_quartets", "onepass_quartets"],
    v=v,
    quartets_root=paired_quartets_root,
    train_val_test_split=True,
    split_by_keys=True,
    transpose=False,
    interleave=interleave,
)

# with h5py.File(paired_dset_path, "r") as f:
#     correct_fnames = sorted(list(f[].keys()))
#     error_fnames = sorted(list(f["train/omr_quartets"].keys()))

#     res = []
#     for pair in [
#         (correct_fnames, "correct_quartets"),
#         (error_fnames, "omr_quartets"),
#         (correct_onepass_fnames, "correct_quartets"),
#         (error_onepass_fnames, "onepass_quartets"),
#     ]:
#         fnames, k = pair
#         res.append([f[k][x][:].astype(np.uint16) for x in fnames])

#     correct_dset, error_dset, error_onepass_dset, correct_onepass_dset = res

correct_path = "train/correct_quartets"
error_path = "train/omr_quartets"

X, Y = get_training_samples(correct_path, error_path, paired_dset_path, bands=0.1)
print("training samples successfully acquired")
X = np.array(X).reshape(-1, 1)
Y = np.array(Y)

np.save("X.npy", X)
np.save("Y.npy", Y)

# X = np.load("X.npy")
# Y = np.load("Y.npy")

num_samples = Y.shape[0]
new_num_samples = min(250000, num_samples)
downsample = np.random.choice(num_samples, new_num_samples, replace=False)

err_gen = ErrorGenerator(labeled_data=[X[downsample], Y[downsample]])
err_gen.save_models(error_generator_fname)

err_gen = ErrorGenerator(error_models_fpath=error_generator_fname)
# now, make .h5 file of test sequences
pms = []
for splt in ["train", "test", "validate"]:
    pms.append((f"{splt}/correct_quartets", f"{splt}/omr_quartets", f"{splt}/omr"))
    pms.append(
        (f"{splt}/correct_quartets", f"{splt}/onepass_quartets", f"{splt}/onepass")
    )

make_supervised_examples(
    pms, paired_dset_path, supervised_targets_fname, err_gen=err_gen, bands=0.15
)
