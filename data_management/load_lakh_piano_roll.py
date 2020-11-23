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
import pypianoroll as ppr
reload(fcts)


lpr_root = r"D:\Documents\datasets\lakh_midi_dataset\lpd_5\lpd_5_cleansed"
all_fnames = []
for root, dirs, files in os.walk(lpr_root):
    for name in files:
        all_fnames.append(os.path.join(root, name))


x = ppr.load(all_fnames[5555])
trs = [t.pianoroll for t in x.tracks if t.pianoroll.shape[0] > 0]
stack_pr = np.stack(trs).clip(0, 1)
reduced_pr = np.sum(stack_pr, 0)

hz_proj = np.sum(reduced_pr, 1).nonzero()[0]
vt_proj = np.sum(reduced_pr, 0).nonzero()[0]
trim_pr = reduced_pr[hz_proj[0]:hz_proj[-1] + 1, vt_proj[0]:vt_proj[-1] + 1]

rl_arr = [trim_pr[0, :]]
deltas = []
sames = 0
last_n = 0
for n in range(1, trim_pr.shape[0]):
    if all(rl_arr[-1] == trim_pr[n, :]):
        sames += 1
        continue
    new_entry = trim_pr[n, :]
    rl_arr.append(new_entry)
    deltas.append(n - last_n)
    last_n = n
rl = np.stack(rl_arr)
deltas += [0]
deltas = np.expand_dims(np.array(deltas), 1)
rl = np.concatenate([deltas, rl], 1)

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
