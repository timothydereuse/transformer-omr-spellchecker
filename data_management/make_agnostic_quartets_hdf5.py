import music21 as m21
import os
from collections import namedtuple, Counter
import numpy as np
import h5py
import data_management.semantic_to_agnostic as sta
import data_management.vocabulary as vocab

test_proportion = 0.2
validate_proportion = 0.1
beat_multiplier = 48
num_transpositions_per_file = 2
possible_transpositions = ["m2", "M2", "m3", "M3", "P4", "a4"]
possible_transpositions = possible_transpositions + [
    "-" + x for x in possible_transpositions
]
# interleave = True
# all_keys = ['ABC', 'kernscores', 'felix', 'felix_errors']
# c = m21.converter.Converter()

# parse all files and build vocabulary, first.
def build_vocab(all_keys, out_fname, quartets_root, interleave=True):
    all_tokens = Counter()
    print("building vocabulary...")
    for k in all_keys:

        print(f"now processing {k}. currently {len(all_tokens)} unique tokens.")
        files = os.listdir(os.path.join(quartets_root, k))
        for fname in files:
            print(f"processing {fname}...")
            fpath = os.path.join(os.path.join(quartets_root, k, fname))
            try:
                parsed_file = m21.converter.parse(fpath)
            except Exception:
                print(f"parsing {fname} failed, skipping file")
                continue
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))
            agnostic = sta.m21_parts_to_interleaved_agnostic(
                parts, remove=["+"], just_tokens=True, interleave=interleave
            )
            all_tokens.update(agnostic)
    v = vocab.Vocabulary(all_tokens)
    v.save_vocabulary(out_fname)


# then parse them again to actually save them.
def make_hdf5(
    dset_path,
    keys,
    v,
    quartets_root,
    train_val_test_split=True,
    split_by_keys=False,
    transpose=False,
    interleave=True,
):
    with h5py.File(dset_path, "a") as f:
        f.attrs["beat_multiplier"] = beat_multiplier

        if train_val_test_split and not split_by_keys:
            train_grp = f.create_group("train")
            test_grp = f.create_group("test")
            validate_grp = f.create_group("validate")
        elif split_by_keys and not train_val_test_split:
            grps = {k: f.create_group(k) for k in keys}
        elif train_val_test_split and split_by_keys:
            train_grp = f.create_group("train")
            test_grp = f.create_group("test")
            validate_grp = f.create_group("validate")
            for g in [train_grp, test_grp, validate_grp]:
                grps = {k: g.create_group(k) for k in keys}

    # get all possible fnames
    all_fnames = set()
    for k in keys:
        fnames = os.listdir(os.path.join(quartets_root, k))
        fnames = ["".join(x.split(".")[:-1]) for x in fnames]
        all_fnames.update(fnames)
    all_fnames = list(all_fnames)
    np.random.shuffle(all_fnames)

    # assign each filename to one of test, train, or validate
    split_test = int(np.round(test_proportion * len(all_fnames)))
    split_val = int(np.round(validate_proportion * len(all_fnames)) + split_test)
    split_classifier = {}
    for i, fn in enumerate(all_fnames):
        if i <= split_test:
            split_classifier[fn] = "test"
        elif i <= split_val:
            split_classifier[fn] = "validate"
        else:
            split_classifier[fn] = "train"

    for k in keys:
        files = os.listdir(os.path.join(quartets_root, k))
        np.random.shuffle(files)

        for i, fname in enumerate(files):
            noext_fname = "".join(fname.split(".")[:-1])
            print(f"parsing {k}/{fname}...")

            fpath = os.path.join(quartets_root, k, fname)
            try:
                parsed_file = m21.converter.parse(fpath)
            except Exception:
                print(f"parsing {k}/{fname} failed, skipping file")
                continue
            parts = list(parsed_file.getElementsByClass(m21.stream.Part))

            if transpose:
                transpositions = np.random.choice(
                    possible_transpositions, num_transpositions_per_file, replace=False
                )
                transpositions = np.concatenate([[None], transpositions])
            else:
                transpositions = [None]

            agnostics = [
                sta.m21_parts_to_interleaved_agnostic(
                    parts,
                    transpose=x,
                    remove=["+"],
                    just_tokens=True,
                    interleave=interleave,
                )
                for x in transpositions
            ]
            agnostic_vecs = [v.words_to_vec(x) for x in agnostics]
            arrs = [np.array(x) for x in agnostic_vecs]
            res.extend(arrs)
            spl_name = split_classifier[noext_fname]
            with h5py.File(dset_path, "a") as f:

                if not train_val_test_split and not split_by_keys:
                    selected_subgrp = f
                elif not train_val_test_split and split_by_keys:
                    selected_subgrp = f[k]
                elif train_val_test_split and not split_by_keys:
                    selected_subgrp = f[spl_name]
                else:
                    selected_subgrp = f[spl_name][k]

                for i, arr in enumerate(arrs):
                    tpose = transpositions[i]
                    name = f"{noext_fname}-tposed.{tpose}"
                    selected_subgrp.create_dataset(
                        name=name, data=arr, compression="gzip"
                    )
