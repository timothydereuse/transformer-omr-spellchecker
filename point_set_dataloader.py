from torch.utils.data import IterableDataset, DataLoader
from importlib import reload
import torch
import os
import numpy as np
import h5py
import logging
import model_params as params

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


class MidiNoteTupleDataset(IterableDataset):

    program_ranges = {
        (0, 24): 0,
        (25, 32): 1,
        (33, 40): 2,
        (41, 128): 3
    }

    def __init__(self, dset_fname, seq_length, num_feats=4, base=None, shuffle_files=True,
                 padding_amt=None, random_offsets=True, estimate_stats_batches=30, dataset_proportion=False,
                 use_stats_from=None):
        """
        @dset_fname - the file name of the processed hdf5 dataset
        @seq_length - length to chop sequences into
        @num_feats - number of features to use - must be 2, 3, or 4. in order of inclusion:
            (pitch, time, duration, voice)
        @base - a subset of the path names within the hdf5 file (optional)
        @shuffle_files - randomizes order of loading songs from hdf5 file (optional)
        @padding_amt - amount of padding to add to beginning and end of each song (optional,
            default: @seq_length // 2)
        @random_offsets - randomize start position of sequences (optional, default: true)
        @estimate_stats_batches - number of batches to use for stat estimation
        @dataset_proportion - set to true to dramatically reduce size of dataset
        """
        super(MidiNoteTupleDataset).__init__()

        self.dset_fname = dset_fname
        self.seq_length = seq_length
        self.num_feats = num_feats
        self.random_offsets = random_offsets
        self.shuffle_files = shuffle_files
        self.dataset_proportion = dataset_proportion
        self.flags = params.notetuple_flags

        self.f = h5py.File(self.dset_fname, 'r')
        if base is not None:
            self.f = self.f[base]
        self.fnames = all_hdf5_keys(self.f)
        if dataset_proportion:
            self.fnames = self.fnames[:int(np.ceil(len(self.fnames) * dataset_proportion))]

        self.num_feats = num_feats
        padding_element = np.array(self.flags['pad'])[:self.num_feats]

        self.padding_amt = padding_amt if padding_amt else self.seq_length // 5
        self.padding_seq = np.stack(
            [padding_element for _ in range(self.padding_amt)], 0)

        self.stds = torch.ones(self.num_feats)
        self.means = torch.zeros(self.num_feats)
        if use_stats_from is not None:
            self.stds, self.means = use_stats_from.stds, use_stats_from.means
        elif estimate_stats_batches > 0:
            self.stds, self.means = self.estimate_stats()

    def simplify_programs(self, programs):
        x = np.zeros(programs.shape)
        for k in self.program_ranges.keys():
            in_category = (k[0] < programs) * (programs <= k[1])
            x[in_category] = self.program_ranges[k]
        return np.expand_dims(x, 1)

    def estimate_stats(self, num_vals=1e6):
        '''
        estimates mean and stdv of each feature using one-pass mean and variance calculation
        '''
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
            # x = self.f[self.fnames[0]]

            # no need to use a custom factorization, just extract the relevant columns
            # onset, duration, time to next onset, pitch, velocity, program
            programs = self.simplify_programs(x[:, 5])
            # notetuples = np.concatenate([x[:, 1:4], programs], 1)
            notetuples = np.concatenate([x[:, [0, 2, 3, 4]], programs], 1)
            notetuples = notetuples[:, :self.num_feats]

            # pad runlength encoding on both sides
            padded_nt = np.concatenate([
                self.padding_seq,
                notetuples,
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
                seq = padded_nt[st:end]

                # recenter each individual batch to start at time = 0
                min_time_offset = np.min(seq[:, 0])
                seq[:, 0] -= min_time_offset

                yield seq


if __name__ == '__main__':
    fname = 'all_string_quartets.h5'
    seq_len = 500
    proportion = 0.2
    dset = MidiNoteTupleDataset(fname, seq_len, num_feats=4, dataset_proportion=1, shuffle_files=False)

    dload = DataLoader(dset, batch_size=15)
    for j in range(10):
        batches = []
        for i, x in enumerate(dload):
            # print(i, x.shape)
            batches.append(x)
        print(i, len(batches))


