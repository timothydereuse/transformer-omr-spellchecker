import h5py
import numpy as np
import data_augmentation.needleman_wunsch_alignment as align
from numba import njit
from collections import Counter
import data_augmentation.error_gen_logistic_regression as elgr


def get_training_samples(correct_dset, error_dset, correct_fnames):
    error_notes = {x: [] for x in ["replace_mod", "insert_mod"]}

    # training samples for logistic regression (MaxEnt Markov Model) for creating errors
    X = []
    Y = []
    for ind in range(len(correct_dset)):

        print(f"aligning {correct_fnames[ind]}...")
        correct_seq = [x for x in correct_dset[ind]]
        error_seq = [x for x in error_dset[ind]]
        correct_align, error_align, r, score = align.perform_alignment(
            correct_seq,
            error_seq,
            match_weights=[3, -2],
            gap_penalties=[-2, -2, -1, -1],
        )

        print("".join(r))

        errors = []

        err_to_class = {"O": 0, "~": 1, "+": 2, "-": 3}
        most_recent_correct_note = 0

        for i in range(len(correct_align)):

            error_note = error_align[i]
            correct_note = correct_align[i]

            # this horrible string hack is necessary beause sklearn's label encoder takes only strings or numbers as input,
            # not tuples. everything is a string of '[error type].[replace/insertion entry]'
            c = err_to_class[r[i]]
            if r[i] == "O":
                label = f"{c}.0"
            elif r[i] == "~":
                error_notes["replace_mod"].append(error_note)
                label = f"{c}.{error_note}"
            elif r[i] == "+":
                error_notes["insert_mod"].append(error_note)
                label = f"{c}.{error_note}"
            elif r[i] == "-":
                label = f"{c}.0"

            if not type(correct_note) == str:
                most_recent_correct_note = correct_note
            # prev_entries = [err_to_class[r[i - x]] if i - x >= 0  else err_to_class[r[0]] for x in range(1, ngram + 1)]
            sample = most_recent_correct_note

            X.append(sample)
            Y.append(label)
    return X, Y


def make_supervised_examples(pms, supervised_targets_fname, err_gen, correct_dset):
    for t in pms:
        target_dset, target_dset_fnames, category = t

        with h5py.File(supervised_targets_fname, "a") as f:
            f.create_group(category)

        for ind in range(len(correct_dset)):
            print(f"making training sequence for {target_dset_fnames[ind]}...")

            correct_seq = correct_dset[ind]
            error_seq = target_dset[ind]

            err, Y = err_gen.add_errors_to_seq(correct_seq, error_seq)
            arr = np.stack([err, Y])

            with h5py.File(supervised_targets_fname, "a") as f:
                name = target_dset_fnames[ind]
                g = f[category]
                dset = g.create_dataset(name=name, data=arr, compression="gzip")


def get_raw_probs(error_generator, v):
    e = error_generator
    res = {}

    for word in v.wtv.keys():
        num = v.wtv[word]
        sharp_probs = e.regression.predict_proba(e.enc.transform([[num]]))[0]
        entry = [
            (
                x,
                sharp_probs[x],
                e.enc_labels.inverse_transform([x])[0][0],
                v.vec_to_words(
                    [int(e.enc_labels.inverse_transform([x])[0].split(".")[-1])]
                ),
            )
            for x in np.argsort(sharp_probs)
        ]
        res[word] = entry

    return res
