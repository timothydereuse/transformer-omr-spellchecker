import json
import pretty_midi as pm
import factorizations as fcts
from importlib import reload
from matplotlib import pyplot as plt
from itertools import product
from mido import KeySignatureError
import os
import torch
import numpy as np
# from scipy.sparse import csc_matrix
reload(fcts)


json_key = r"D:\Documents\lakh_midi_dataset\md5_to_paths.json"
lmd_root = r"D:\Documents\lakh_midi_dataset\lmd_full"


def get_midi_from_md5(md5):
    return pm.PrettyMIDI(rf'{lmd_root}/{md5[0]}/{md5}.mid')


def load_lmd_random(num=10):
    with open(json_key) as f:
        j = json.load(f)
    md5s = list(j.keys())

    pms = []
    while len(pms) < num:
        ind = np.random.randint(len(md5s))
        try:
            mid = get_midi_from_md5(md5s[ind])
        except OSError:
            continue
        except EOFError:
            continue
        except KeySignatureError:
            continue
        except ValueError:
            continue
        except IndexError:
            continue
        pms.append(mid)

    return pms


def load_lmd_runlength(num, seq_length):

    with open(json_key) as f:
        j = json.load(f)
    md5s = list(j.keys())

    inds = np.random.choice(range(len(md5s)), num, False)

    runlengths = []
    for ind in inds:
        try:
            mid = get_midi_from_md5(md5s[ind])
        except OSError:
            continue
        except EOFError:
            continue
        except KeySignatureError:
            continue
        except ValueError:
            continue
        runlength = (fcts.pm_to_runlength(mid))

        num_chunks = runlength.shape[0] // seq_length
        if not num_chunks:
            continue
        rl_split = np.array_split(runlength[:num_chunks * seq_length], num_chunks)
        runlengths += rl_split[:-1]

    full_set = torch.Tensor(np.stack(runlengths))

    # put it into S N E format, as pytorch needs it
    full_set = torch.transpose(full_set, 0, 1)

    # remove time for simple testing
    full_set = full_set[:, :, :-1]

    return full_set


def load_lmd_note_tuple(num, seq_length):
    pms = load_lmd_random(num)

    all_songs = []
    for pm in pms:
        tuples = fcts.pm_to_note_tuple(pm)
        num_chunks = tuples.shape[0] // seq_length
        if not num_chunks:
            continue
        split = np.array_split(tuples[:num_chunks * seq_length], num_chunks)
        all_songs += split[:-1]

    full_set = torch.Tensor(np.stack(all_songs))

    # put it into S N E format, as pytorch needs it
    full_set = torch.transpose(full_set, 0, 1)

    return full_set


def load_mtc_notetuple(num=10, seq_length=20):
    mids_path = "D:\Desktop\meertens_tune_collection\mtc-fs-1.0.tar\midi"
    midi_fnames = os.listdir(mids_path)

    choose_fnames = np.random.choice(midi_fnames, num, False)
    notetuples = []
    for mid_name in choose_fnames:
        x = pm.PrettyMIDI(f"{mids_path}/{mid_name}")
        notetuples.append(fcts.pm_to_note_tuple(x))

    x = chunk_seqs(notetuples, seq_length)

    return torch.transpose(torch.tensor(x), 0, 1)


def load_mtc_runlength(delta_mapping, pitch_range, num=10, seq_length=20):
    mids_path = r"D:\Desktop\meertens_tune_collection\mtc-fs-1.0.tar\midi"
    midi_fnames = os.listdir(mids_path)

    choose_fnames = np.random.choice(midi_fnames, num, False)
    notetuples = []
    for mid_name in choose_fnames:
        x = pm.PrettyMIDI(f"{mids_path}/{mid_name}")
        notetuples.append(fcts.pm_to_runlength(x, delta_mapping, pitch_range, monophonic=True))

    x = chunk_seqs(notetuples, seq_length)

    return torch.transpose(torch.tensor(x), 0, 1)


def chunk_seqs(seqs, seq_length):
    res = []
    for seq in seqs:
        num_chunks = seq.shape[0] // seq_length
        if not num_chunks:
            continue
        split = np.split(seq[:num_chunks * seq_length], num_chunks)
        res += split

        # add rest with zero-padding if the remainder is long enough
        remainder = seq[num_chunks * seq_length:]
        if len(remainder) < (0.7 * seq_length):
            continue

        padding = np.zeros((seq_length,) + remainder[0].shape)
        padding[:len(remainder)] = remainder
        res.append(padding)

    return np.stack(res)


def get_class_weights(inp):
    class_sums = torch.sum(inp, (0, 1))
    class_sums = (class_sums + 1) / max(class_sums)
    weights = 1 / (class_sums)
    return weights



if __name__ == '__main__':
    def compare_piano_roll_tokens(ta, tb):
        anded = np.dot(ta, tb)
        return anded

    # mids = load_lmd_runlength
    #
    # pr = mid.get_piano_roll(1 / mid.tick_to_time(12))
    # pr = np.clip(pr, 0, 1)
    # # ssm = np.zeros((pr.shape[1], pr.shape[1]))
    # # for x, y in product(range(ssm.shape[0]), range(ssm.shape[1])):
    # #     if y > x:
    # #         continue
    # #     ssm[x, y] = compare_piano_roll_tokens(pr[:, x], pr[:, y])
    #
    # ssm2 = np.matmul(pr.T, pr)
    #
    # ssm_rl = np.matmul(runlength[:, :-1], runlength[:, :-1].T)
    #
    # plt.clf()
    # plt.imshow(ssm_rl)
    # plt.show()
    #
    # plt.clf()
    # plt.imshow(ssm2)
    # plt.show()
