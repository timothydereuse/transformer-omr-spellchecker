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
                 padding_amt=None, random_offsets=True, all_subsequences=False, dataset_proportion=False):
        """
        @dset_fname - the file name of the processed hdf5 dataset
        @seq_length - length to chop sequences into
        @base - a subset of the path names within the hdf5 file (optional)
        @shuffle_files - randomizes order of loading songs from hdf5 file (optional)
        @padding_amt - amount of padding to add to beginning and end of each song (optional,
            default: @seq_length // 2)
        @random_offsets - randomize start position of sequences (optional, default: true)
        @all_subsequences - take all overlapping subsequences of all inputs, instead of
            cutting them into non-overlapping segments
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
        self.all_subsequences = all_subsequences

        self.f = h5py.File(self.dset_fname, 'r')
        if base is not None:
            self.f = self.f[base]
        self.fnames = all_hdf5_keys(self.f)
        if dataset_proportion:
            self.fnames = self.fnames[:int(np.ceil(len(self.fnames) * dataset_proportion))]

        self.padding_amt = padding_amt if padding_amt else self.seq_length // 5
        self.padding_seq = np.zeros(self.padding_amt, dtype=np.float32) + self.vocabulary.SEQ_PAD
        self.target_padding_seq = np.zeros(self.padding_amt, dtype=np.float32)

    def __iter__(self):
        '''
        Main iteration function.
        '''

        if self.shuffle_files:
            np.random.shuffle(self.fnames)

        # iterate through all given fnames, breaking them into chunks of seq_length...
        for fname in self.fnames:
            glyphs = self.f[fname]

            # determine if this is raw input for data augmentation or data along with targets
            with_targets = (len(glyphs.shape) > 1)

            # pad on both sides
            if not with_targets:
                padded_glyphs = np.concatenate([
                    self.padding_seq,
                    [self.vocabulary.SEQ_SOS],
                    glyphs,
                    [self.vocabulary.SEQ_EOS],
                    self.padding_seq
                    ])
            else:
                arrs = [
                    np.stack([self.padding_seq, self.target_padding_seq]),
                    np.expand_dims([self.vocabulary.SEQ_SOS, 0], 1),
                    glyphs,
                    np.expand_dims([self.vocabulary.SEQ_EOS, 0], 1),
                    np.stack([self.padding_seq, self.target_padding_seq])
                    ]
                padded_glyphs = np.concatenate(arrs, 1)             

            padded_length = padded_glyphs.shape[1] if with_targets else padded_glyphs.shape[0]
            # figure out how many sequences we can get out of this
            if not self.all_subsequences:
                num_seqs = np.floor(padded_length / self.seq_length)
                remainder = padded_length - (num_seqs * self.seq_length)
                offset = np.random.randint(remainder + 1) if self.random_offsets else 0
            else:
                num_seqs = (padded_length - self.seq_length)
                offset = 0

            # check if the current file is too short to be used with the seq_length desired
            if num_seqs == 0:
                continue

            # return sequences of notes from each file, seq_length in length.
            # move to the next file when the current one has been exhausted.
            for i in range(int(num_seqs)):
                if not self.all_subsequences:
                    st = i * self.seq_length + offset
                    end = (i+1) * self.seq_length + offset
                else:
                    st = i
                    end = i + self.seq_length
                seq = (padded_glyphs[0, st:end], padded_glyphs[1, st:end]) if with_targets else padded_glyphs[st:end] 
                yield seq, (f'{fname}-{i}')


if __name__ == '__main__':
    from data_management.vocabulary import Vocabulary
    fname = 'processed_datasets/all_string_quartets_agnostic.h5'
    seq_len = 500
    proportion = 0.02
    v = Vocabulary(load_from_file='./data_management/vocab.txt')
    dset = AgnosticOMRDataset(fname, seq_len, v, dataset_proportion=0.5, shuffle_files=False, all_subsequences=True)

    dload = DataLoader(dset, batch_size=15)
    for j in range(1):
        batches = []
        for i, x in enumerate(dload):
            print(i, x[0].shape)
            batches.append(x)
        print(i, len(batches))

    fname = 'processed_datasets/supervised_omr_targets.h5'
    dset = AgnosticOMRDataset(fname, seq_len, v, dataset_proportion=1, shuffle_files=False)

    dload = DataLoader(dset, batch_size=15)

    batches = []
    for i, x in enumerate(dload):
        print(i, len(x[0]), len(x[1]))
        batches.append(x)
    print(i, len(batches))



