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


class AgnosticOMRDataset(IterableDataset):

    def __init__(self, dset_fname, seq_length, vocabulary, base=None, shuffle_files=True,
                 padding_amt=None, random_offsets=True, dataset_proportion=False,
                 use_stats_from=None):
        """
        @dset_fname - the file name of the processed hdf5 dataset
        @seq_length - length to chop sequences into
        @base - a subset of the path names within the hdf5 file (optional)
        @shuffle_files - randomizes order of loading songs from hdf5 file (optional)
        @padding_amt - amount of padding to add to beginning and end of each song (optional,
            default: @seq_length // 2)
        @random_offsets - randomize start position of sequences (optional, default: true)
        @dataset_proportion - set to true to dramatically reduce size of dataset
        """
        super(AgnosticOMRDataset).__init__()

        self.dset_fname = dset_fname
        self.seq_length = seq_length
        self.random_offsets = random_offsets
        self.shuffle_files = shuffle_files
        self.dataset_proportion = dataset_proportion
        self.flags = params.notetuple_flags
        self.vocabulary = vocabulary

        self.f = h5py.File(self.dset_fname, 'r')
        if base is not None:
            self.f = self.f[base]
        self.fnames = all_hdf5_keys(self.f)
        if dataset_proportion:
            self.fnames = self.fnames[:int(np.ceil(len(self.fnames) * dataset_proportion))]

        self.padding_amt = padding_amt if padding_amt else self.seq_length // 5
        self.padding_seq = np.zeros(self.padding_amt, dtype=np.float32) + self.vocabulary.SEQ_PAD

    def __iter__(self):
        '''
        Main iteration function.
        '''

        if self.shuffle_files:
            np.random.shuffle(self.fnames)

        # iterate through all given fnames, breaking them into chunks of seq_length...
        for fname in self.fnames:
            x = self.f[fname]
            glyphs = x

            # pad runlength encoding on both sides
            padded_glyphs = np.concatenate([
                self.padding_seq,
                glyphs,
                self.padding_seq
                ])

            # figure out how many sequences we can get out of this
            num_seqs = np.floor(padded_glyphs.shape[0] / self.seq_length)

            # check if the current file is too short to be used with the seq_length desired
            if num_seqs == 0:
                continue

            remainder = padded_glyphs.shape[0] - (num_seqs * self.seq_length)
            offset = np.random.randint(remainder + 1) if self.random_offsets else 0

            # return sequences of notes from each file, seq_length in length.
            # move to the next file when the current one has been exhausted.
            for i in range(int(num_seqs)):
                st = i * self.seq_length + offset
                end = (i+1) * self.seq_length + offset
                seq = padded_glyphs[st:end]

                yield seq


if __name__ == '__main__':
    from data_management.vocabulary import Vocabulary
    fname = 'all_string_quartets_agnostic.h5'
    seq_len = 500
    proportion = 0.2
    v = Vocabulary(load_from_file='./data_management/vocab.txt')
    dset = AgnosticOMRDataset(fname, seq_len, v, dataset_proportion=1, shuffle_files=False)

    dload = DataLoader(dset, batch_size=15)
    for j in range(10):
        batches = []
        for i, x in enumerate(dload):
            # print(i, x.shape)
            batches.append(x)
        print(i, len(batches))


