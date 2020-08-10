import json
from torch.utils.data import IterableDataset, DataLoader
import pretty_midi as pm
import factorizations as fcts
from importlib import reload
from collections import Counter
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
