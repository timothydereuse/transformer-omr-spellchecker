from torch.utils.data import IterableDataset, DataLoader
import pretty_midi as pm
import factorizations as fcts
from importlib import reload
from collections import Counter
import os
import torch
import numpy as np
reload(fcts)


def get_tick_deltas_for_runlength(mids_path, midi_fnames=None, num_dur_vals=16, proportion=0.2):

    if midi_fnames is None:
        midi_fnames = os.listdir(mids_path)
    num_choice = int(proportion * len(midi_fnames))
    midi_fnames = np.random.choice(midi_fnames, num_choice, replace=False)

    c = Counter()
    pitch_c = Counter()

    for i, fname in enumerate(midi_fnames):
        pm_file = pm.PrettyMIDI(f"{mids_path}/{fname}")
        all_notes = []
        for voice in pm_file.instruments:
            if voice.is_drum:
                continue
            all_starts = [pm_file.time_to_tick(n.start) for n in voice.notes]
            all_notes += all_starts

            all_pitches = [n.pitch for n in voice.notes]
            pitch_c.update(all_pitches)

        diffs = np.diff(all_starts)
        c.update(diffs)

        if not i % 200:
            print(f"processing tick deltas: {i} of {len(midi_fnames)}")

    top_durs = c.most_common(num_dur_vals)

    most = np.sort([x[0] for x in top_durs])
    res_dict = {v: i for i, v in enumerate(most)}

    pitch_range = (min(pitch_c.keys()), max(pitch_c.keys()))

    return res_dict, pitch_range


def load_mtc_notetuple(num=10, seq_length=20):
    mids_path = r"D:\Desktop\meertens_tune_collection\mtc-fs-1.0.tar\midi"
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

    def __init__(self, root_dir, seq_length, fnames=None, num_dur_vals=None, use_stats_from=None, proportion_for_stats=0.3, shuffle_files=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(MTCDataset).__init__()
        self.root_dir = root_dir
        self.seq_length = seq_length

        if fnames is None:
            self.midi_fnames = os.listdir(self.root_dir)
        else:
            self.midi_fnames = fnames

        if shuffle_files:
            np.random.shuffle(self.midi_fnames)

        if use_stats_from is None:
            assert num_dur_vals is not None, "one of num_dur_vals OR use_stats_from must be supplied"
            dmap, prange = get_tick_deltas_for_runlength(
                self.root_dir, fnames, num_dur_vals, proportion_for_stats)
            self.delta_mapping = dmap
            self.pitch_range = prange
            self.num_dur_vals = num_dur_vals
        else:
            self.delta_mapping = use_stats_from.delta_mapping
            self.pitch_range = use_stats_from.pitch_range
            self.num_dur_vals = use_stats_from.num_dur_vals

        self.num_feats = self.pitch_range[1] - self.pitch_range[0] + self.num_dur_vals + 1

        padding_element = np.zeros(self.num_feats)
        padding_element[-1] = 1
        padding_element[0] = 1
        self.padding_seq = np.stack(
            [padding_element for _ in range(self.seq_length + 1)],
            0)

        self.pitch_weights, self.dur_weights = self.get_weights(proportion_for_stats)

    def __iter__(self, fnames=None):

        if fnames is None:
            fnames = self.midi_fnames

        for fname in self.midi_fnames:
            x = pm.PrettyMIDI(f"{self.root_dir}/{fname}")
            runlength = fcts.pm_to_runlength(
                x, self.delta_mapping, self.pitch_range, monophonic=True)
            num_seqs = np.round(runlength.shape[0] / self.seq_length)

            pad_rl = np.concatenate([runlength, self.padding_seq])
            for i in range(int(num_seqs)):
                slice = pad_rl[i * self.seq_length:(i+1) * self.seq_length]
                yield slice

    def get_weights(self, proportion=0.4):
        num = int(len(self.midi_fnames) * proportion)
        fnames = np.random.choice(self.midi_fnames, num, replace=False)

        sums = torch.zeros(self.num_feats)
        for batch in self.__iter__(fnames):
            sums += batch.reshape(-1, self.num_feats).sum(0)
        sums = sums.max() / (sums + 1)
        pitch = sums[:-self.num_dur_vals].float()
        dur = sums[-self.num_dur_vals:].float()
        return pitch, dur

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
