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


def get_tick_deltas_for_runlength(mids_path, num_dur_vals=16, proportion=0.2):
    midi_fnames = os.listdir(mids_path)
    num_choice = int(proportion * len(midi_fnames))
    midi_fnames = np.random.choice(midi_fnames, num_choice, replace=False)

    c = Counter()

    min_pitches = []
    max_pitches = []

    for i, fname in enumerate(midi_fnames):
        pm_file = pm.PrettyMIDI(f"{mids_path}/{fname}")
        all_notes = []
        for voice in pm_file.instruments:
            if voice.is_drum:
                continue
            all_starts = [pm_file.time_to_tick(n.start) for n in voice.notes]
            all_notes += all_starts

            min_pitches.append(min([n.pitch for n in voice.notes]))
            max_pitches.append(max([n.pitch for n in voice.notes]))

        diffs = np.diff(all_starts)
        c.update(diffs)

        if not i % 200:
            print(f"processing tick deltas: {i} of {len(midi_fnames)}")

    most = [x[0] for x in c.most_common(num_dur_vals)]
    most = np.sort(most)
    res_dict = {v: i for i, v in enumerate(most)}

    pitch_range = (min(min_pitches), max(max_pitches))

    return res_dict, pitch_range


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


class MTCDataset(IterableDataset):
    """meertens_tune_collection dataset."""

    def __init__(self, root_dir, seq_length, num_dur_vals=17, proportion=0.2, shuffle_files=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(MTCDataset).__init__()
        self.root_dir = root_dir
        self.midi_fnames = os.listdir(self.root_dir)
        self.seq_length = seq_length

        if shuffle_files:
            np.random.shuffle(self.midi_fnames)

        dmap, prange = get_tick_deltas_for_runlength(
            self.root_dir, num_dur_vals, proportion)
        self.delta_mapping = dmap
        self.pitch_range = prange
        self.num_dur_vals = num_dur_vals

        self.num_feats = self.pitch_range[1] - self.pitch_range[0] + self.num_dur_vals + 1
        padding_element = np.zeros(self.num_feats)
        padding_element[-1] = 1
        padding_element[0] = 1
        self.padding_seq = np.stack(
            [padding_element for _ in range(self.seq_length + 1)],
            0)

    def __iter__(self):
        for fname in self.midi_fnames:
            x = pm.PrettyMIDI(f"{self.root_dir}/{fname}")
            runlength = fcts.pm_to_runlength(
                x, self.delta_mapping, self.pitch_range, monophonic=True)
            num_seqs = np.ceil(runlength.shape[0] / self.seq_length)

            pad_rl = np.concatenate([runlength, self.padding_seq])
            for i in range(int(num_seqs)):
                slice = pad_rl[i * self.seq_length:(i+1) * self.seq_length]
                yield slice


if __name__ == '__main__':
    root_dir = r"D:\Desktop\meertens_tune_collection\mtc-fs-1.0.tar\midi"
    num_dur_vals = 17
    seq_len = 30
    proportion = 0.2
    dset = MTCDataset(root_dir, seq_len, num_dur_vals, proportion)

    dload = DataLoader(dset, batch_size=20)
    for i, x in enumerate(dload):
        print(i, x.shape)
        if i > 2:
            break
