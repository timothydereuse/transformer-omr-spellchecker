from torch.utils.data import IterableDataset, DataLoader
import factorizations as fcts
from importlib import reload
from collections import Counter
import torch
import numpy as np
import h5py
import logging
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


def get_class_weights(inp):
    class_sums = torch.sum(inp, (0, 1))
    class_sums = (class_sums + 1) / max(class_sums)
    weights = 1 / (class_sums)
    return weights


def get_all_hdf5_fnames(f, base=None):
    fnames = []
    for k in f.keys():
        for n in f[k].keys():
            fnames.append(rf'{k}/{n}')
    return fnames


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
                 random_offsets=True, trial_run=False):
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

        # self.flags = {
        #     ''
        # }

        self.f = h5py.File(self.dset_fname, 'r')
        if base is not None:
            self.f = self.f[base]
        self.fnames = all_hdf5_keys(self.f)
        if trial_run:
            self.fnames = self.fnames[:100]

        self.num_dur_vals = num_dur_vals

        if use_stats is None:
            dmap, prange = get_tick_deltas_for_runlength(
                self.f, self.fnames, num_dur_vals, proportion_for_stats)
            self.delta_mapping = dmap
            self.pitch_range = prange
        else:
            self.pitch_range = use_stats[0]
            self.delta_mapping = use_stats[1]

        # the number of features is this sum plus 3 (one for an off-by-one error caused by
        # the pitch range being inclusive, one for the 'rest' message, one for 'fill in' message)
        self.num_feats = self.pitch_range[1] - self.pitch_range[0] + self.num_dur_vals + 2

        padding_element = np.zeros(self.num_feats)
        padding_element[-1] = 1
        padding_element[0] = 1
        self.padding_amt = padding_amt if padding_amt else self.seq_length // 2
        self.padding_seq = np.stack(
            [padding_element for _ in range(self.padding_amt)], 0)

        # self.pitch_weights, self.dur_weights = self.get_weights(proportion_for_stats)

    def get_stats(self):
        return (self.pitch_range, self.delta_mapping)

    def __iter__(self):
        '''
        Main iteration function.
        @fnames: takes a subset of the fnames in the supplied hdf5 file. will only iterate thru
            the suppied names.
        '''

        if self.shuffle_files:
            np.random.shuffle(self.fnames)

        # iterate through all given fnames, breaking them into chunks of seq_length...
        for fname in self.fnames:
            x = self.f[fname]
            runlength = fcts.arr_to_runlength_mono(
                x, self.delta_mapping, self.pitch_range)

            # pad runlength encoding on both sides
            padded_rl = np.concatenate([self.padding_seq, runlength, self.padding_seq])
            num_seqs = np.floor(padded_rl.shape[0] / self.seq_length)
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


if __name__ == '__main__':
    fname = 'essen_meertens_songs.hdf5'
    num_dur_vals = 10
    seq_len = 30
    proportion = 0.2
    dset = MonoFolkSongDataset(fname, seq_len, num_dur_vals=num_dur_vals,
                      proportion_for_stats=proportion)

    dload = DataLoader(dset, batch_size=15)
    for i, x in enumerate(dload):
        print(i, x.shape)
        if i > 2:
            break
