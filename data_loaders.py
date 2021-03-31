from torch.utils.data import IterableDataset, DataLoader
import factorizations as fcts
from importlib import reload
from collections import Counter
import torch
import os
import numpy as np
import h5py
import logging
import model_params as params
reload(fcts)


def get_tick_deltas_for_runlength(f, fnames, num_dur_vals=16, proportion=0.5):

    num_choice = int(proportion * len(fnames))
    fnames = np.random.choice(fnames, num_choice, replace=False)

    c = Counter()
    pitch_c = Counter()

    for i, fname in enumerate(fnames):
        arr = f[fname]
        all_starts = arr[:, 1]
        all_pitches = [x for x in arr[:, 0] if x > 0]
        pitch_c.update(all_pitches)

        diffs = np.diff(all_starts)
        c.update(diffs)

        if not i % 2000:
            logging.info(f"processing tick deltas: {i} of {len(fnames)}")

    top_durs = c.most_common(num_dur_vals)
    most = np.sort([x[0] for x in top_durs])

    if len(most) < num_dur_vals:
        logging.warn(f'requested number of duration values {num_dur_vals} is too large for actual '
                     f'number of duration values found: {len(most)}. the resulting encoding of this input '
                     f'will have one or more features that are always set to zero.')
        filler = np.arange(0, num_dur_vals - len(most)) + max(most) + 1
        most = np.concatenate([most, filler])

    # deltas = {v: i for i, v in enumerate(most)}
    deltas = most

    pitch_range = (min(pitch_c.keys()), max(pitch_c.keys()))

    return deltas, pitch_range


def get_pitch_range_no_duration(f, fnames, proportion=0.5):
    num_choice = int(proportion * len(fnames))
    fnames = np.random.choice(fnames, num_choice, replace=False)
    pitch_c = Counter()

    for i, fname in enumerate(fnames):
        arr = f[fname]
        all_pitches = [x for x in arr if x > 0]
        pitch_c.update(all_pitches)

    pitch_range = (min(pitch_c.keys()), max(pitch_c.keys()))
    return pitch_range


def all_hdf5_keys(obj):
    '''
    Recursively find all hdf5 keys subordinate to the given object @obj, corresponding to datasets.
    '''
    name_list = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            name_list.append(name)
        else:
            pass

    obj.visititems(visitor_func)
    return name_list


class MonoFolkSongDataset(IterableDataset):
    """meertens_tune_collection dataset."""

    def __init__(self, dset_fname, seq_length, num_dur_vals, base=None, use_stats=None,
                 proportion_for_stats=0.5, shuffle_files=True, padding_amt=None,
                 random_offsets=True, random_transpose=6, trial_run=False):
        """
        @dset_fname - path to hdf5 file created by make_hdf5.py
        @seq_length - number of units to chop sequences into
        @fnames - a subset of the path names within the hdf5 file (optional)
        @num_dur_vals - the number of most common duration values to calculate (optional)
        @use_stats - input another MonoFolkSongDataset's get_stats() into this argument to use its
            calculated stats for generating training batches (optional)
        @proportion_for_stats - in (0, 1]. calculates stats on a random subset of the dataset
            in the hdf5 file (optional)
        @shuffle_files - randomizes order of loading songs from hdf5 file (optional)
        @padding_amt - amount of padding to add to beginning and end of each song (optional,
            default: @seq_length // 2)
        @random_offsets - randomize start position of sequences (optional, default: true)
        @trial_run - set to true to dramatically reduce size of dataset
        """
        super(MonoFolkSongDataset).__init__()
        self.dset_fname = dset_fname
        self.seq_length = seq_length
        self.random_offsets = random_offsets
        self.shuffle_files = shuffle_files
        self.trial_run = trial_run
        self.random_transpose = random_transpose

        self.flags = params.flags

        self.f = h5py.File(self.dset_fname, 'r')
        if base is not None:
            self.f = self.f[base]
        self.fnames = all_hdf5_keys(self.f)
        if trial_run:
            self.fnames = self.fnames[:100]

        self.num_dur_vals = num_dur_vals
        self.use_duration = (num_dur_vals > 0)

        # make sure shape of data makes sense with parameters:
        expected_shape = 2 if self.use_duration else 1
        real_shape = len(self.f[self.fnames[0]].shape)
        assert real_shape == expected_shape, \
            f"num_dur_vals = {num_dur_vals} but dimensionality of data is {real_shape}, not {expected_shape}"

        if use_stats is not None:
            self.pitch_range = use_stats[0]
            self.delta_mapping = use_stats[1]
        elif self.use_duration:
            dmap, prange = get_tick_deltas_for_runlength(
                self.f, self.fnames, num_dur_vals, proportion_for_stats)
            self.delta_mapping = dmap
            self.pitch_range = prange
        else:
            self.pitch_range = get_pitch_range_no_duration(self.f, self.fnames, proportion_for_stats)
            self.delta_mapping = None

        if self.use_duration:
            # the number of features is this sum plus 2 (one for an off-by-one error caused by
            # the pitch range being inclusive, one for the 'rest' message)
            self.pitch_subvector_len = self.pitch_range[1] - self.pitch_range[0] + 2
            self.dur_subvector_len = self.num_dur_vals + len(self.flags)
        else:
            self.pitch_subvector_len = self.pitch_range[1] - self.pitch_range[0] + 2 + len(self.flags)
            self.dur_subvector_len = 0

        self.num_feats = self.pitch_subvector_len + self.dur_subvector_len

        padding_element = np.zeros([self.num_feats])
        padding_element[self.flags['pad']] = 1

        self.padding_amt = padding_amt if padding_amt else self.seq_length // 2
        self.padding_seq = np.stack(
            [padding_element for _ in range(self.padding_amt)], 0)

        self.sos_element = np.zeros([1, self.num_feats])
        self.sos_element[0][self.flags['sos']] = 1

        self.eos_element = np.zeros([1, self.num_feats])
        self.eos_element[0][self.flags['eos']] = 1

        # self.pitch_weights, self.dur_weights = self.get_weights(proportion_for_stats)

    def get_stats(self):
        return (self.pitch_range, self.delta_mapping)

    def apply_transposition(self, x):
        if not self.random_transpose:
            return x
        cx = x[:]
        real_pitches = cx[:, 0][cx[:, 0] > 0]
        upper_leeway = self.pitch_range[1] - max(real_pitches)
        lower_leeway = min(real_pitches) - self.pitch_range[0]
        upper_leeway = min(self.random_transpose, upper_leeway)
        lower_leeway = min(self.random_transpose, lower_leeway)
        transp_amt = np.random.randint(-lower_leeway, upper_leeway)
        cx[:, 0][cx[:, 0] > 0] += transp_amt
        return cx

    def __iter__(self):
        '''
        Main iteration function.
        '''

        if self.shuffle_files:
            np.random.shuffle(self.fnames)

        # iterate through all given fnames, breaking them into chunks of seq_length...
        for fname in self.fnames:
            x = self.f[fname]

            # x = self.apply_transposition(x)
            runlength = fcts.arr_to_runlength_mono(
                x, self.delta_mapping, self.pitch_range, self.flags, self.use_duration)

            # pad runlength encoding on both sides
            padded_rl = np.concatenate([
                self.padding_seq,
                self.sos_element,
                runlength,
                self.eos_element,
                self.padding_seq
                ])

            # figure out how many sequences we can get out of this
            num_seqs = np.floor(padded_rl.shape[0] / self.seq_length)

            # check if the current file is too short to be used with the seq_length desired
            if num_seqs == 0:
                continue

            remainder = padded_rl.shape[0] - (num_seqs * self.seq_length)
            offset = np.random.randint(remainder + 1) if self.random_offsets else 0

            # return sequences of notes from each file, seq_length in length.
            # move to the next file when the current one has been exhausted.
            for i in range(int(num_seqs)):
                st = i * self.seq_length + offset
                end = (i+1) * self.seq_length + offset
                yield padded_rl[st:end]

    def get_weights(self, proportion=0.4):
        '''
        calculates per-category weights over a proportion of the dataset to unbalance classes.
        this is optional - doesn't seem to help training all that much.
        '''
        num = int(len(self.midi_fnames) * proportion)
        fnames = np.random.choice(self.midi_fnames, num, replace=False)

        sums = torch.zeros(self.num_feats)
        for batch in self.__iter__(fnames):
            sums += batch.reshape(-1, self.num_feats).sum(0)
        sums = sums.max() / (sums + 1)
        pitch = sums[:-self.num_dur_vals].float()
        dur = sums[-self.num_dur_vals:].float()
        return pitch, dur


class MidiNoteTupleDataset(IterableDataset):

    program_ranges = {
        (0, 24): 0,
        (25, 32): 1,
        (33, 40): 2,
        (41, 128): 3
    }

    def __init__(self, dset_fname, seq_length, num_feats=4, base=None, shuffle_files=True,
                 padding_amt=None, random_offsets=True, estimate_stats_batches=30, trial_run=False):
        """
        @dset_root -
        @seq_length - number of units to chop sequences into
        @base - a subset of the path names within the hdf5 file (optional)
        @shuffle_files - randomizes order of loading songs from hdf5 file (optional)
        @padding_amt - amount of padding to add to beginning and end of each song (optional,
            default: @seq_length // 2)
        @random_offsets - randomize start position of sequences (optional, default: true)
        @trial_run - set to true to dramatically reduce size of dataset
        """
        super(MidiNoteTupleDataset).__init__()

        self.dset_fname = dset_fname
        self.seq_length = seq_length
        self.random_offsets = random_offsets
        self.shuffle_files = shuffle_files
        self.trial_run = trial_run
        self.flags = params.notetuple_flags

        self.f = h5py.File(self.dset_fname, 'r')
        if base is not None:
            self.f = self.f[base]
        self.fnames = all_hdf5_keys(self.f)
        if trial_run:
            self.fnames = self.fnames[:int(len(self.fnames) * trial_run)]

        self.num_feats = num_feats
        padding_element = np.array(self.flags['pad'])

        self.padding_amt = padding_amt if padding_amt else self.seq_length // 5
        self.padding_seq = np.stack(
            [padding_element for _ in range(self.padding_amt)], 0)

        self.sos_element = np.array([self.flags['sos']])
        self.eos_element = np.array([self.flags['eos']])

        self.stds = torch.ones(self.num_feats)
        self.means = torch.zeros(self.num_feats)
        if estimate_stats_batches > 0:
            self.stds, self.means = self.estimate_stats()

    def simplify_programs(self, programs):
        x = np.zeros(programs.shape)
        for k in self.program_ranges.keys():
            in_category = (k[0] < programs) * (programs <= k[1])
            x[in_category] = self.program_ranges[k]
        return np.expand_dims(x, 1)

    def estimate_stats(self, num_vals=1e6):
        M = 0
        S = 0
        k = 0
        for i, seq in enumerate(self.__iter__()):
            if k > num_vals:
                break
            for idx in range(seq.shape[0]):
                k += 1
                x = seq[idx]
                oldM = M
                M = M + (x - M) / k
                S = S + (x - M) * (x - oldM)
        means = M
        stds = np.sqrt(S / (k-1))
        return torch.tensor(stds, dtype=torch.float), torch.tensor(means, dtype=torch.float)

    def normalize_batch(self, item):
        return (item - self.means) / self.stds

    def unnormalize_batch(self, item):
        return ((item * self.stds) + self.means).round()

    def __iter__(self):
        '''
        Main iteration function.
        '''

        if self.shuffle_files:
            np.random.shuffle(self.fnames)

        # iterate through all given fnames, breaking them into chunks of seq_length...
        for fname in self.fnames:
            x = self.f[fname]

            # no need to use a custom factorization, just extract the relevant columns
            # onset, duration, time to next onset, pitch, velocity, program
            programs = self.simplify_programs(x[:, 5])
            # notetuples = np.concatenate([x[:, 1:4], programs], 1)
            notetuples = np.concatenate([x[:, [0, 1, 3]], programs], 1)

            # pad runlength encoding on both sides
            padded_nt = np.concatenate([
                self.padding_seq,
                self.sos_element,
                notetuples,
                self.eos_element,
                self.padding_seq
                ])

            # figure out how many sequences we can get out of this
            num_seqs = np.floor(padded_nt.shape[0] / self.seq_length)

            # check if the current file is too short to be used with the seq_length desired
            if num_seqs == 0:
                continue

            remainder = padded_nt.shape[0] - (num_seqs * self.seq_length)
            offset = np.random.randint(remainder + 1) if self.random_offsets else 0

            # return sequences of notes from each file, seq_length in length.
            # move to the next file when the current one has been exhausted.
            for i in range(int(num_seqs)):
                st = i * self.seq_length + offset
                end = (i+1) * self.seq_length + offset
                yield padded_nt[st:end]


if __name__ == '__main__':
    fname = 'lmd_cleansed.hdf5'
    num_dur_vals = 0
    seq_len = 500
    proportion = 0.2
    dset = MidiNoteTupleDataset(fname, seq_len, estimate_stats_batches=10)

    dload = DataLoader(dset, batch_size=15)
    batches = []
    for i, x in enumerate(dload):
        print(i, x.shape)
        batches.append(x)
        if i > 2:
            break
