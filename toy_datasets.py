from torch.utils.data import Dataset
import torch
import numpy as np


class SequenceCopyDataset(Dataset):

    def __init__(self, num_feats, num_seqs, seq_length, seq_period, freq_vary=0.2, seq_type='sine'):
        """
        @dset_fname - path to hdf5 file created by make_hdf5.py
        """
        super(SequenceCopyDataset).__init__()
        self.seq_length = seq_length
        self.seq_type = seq_type        # one of 'random', 'sine'
        self.seq_period = seq_period
        self.num_feats = num_feats
        self.num_seqs = num_seqs

        data = np.zeros([num_seqs, seq_length, num_feats])
        slope = np.linspace(1, seq_length, seq_length)
        data[:] = np.tile(slope, [num_feats, 1]).swapaxes(0, 1)

        additive = np.random.normal(0, 10, [num_seqs, num_feats])
        additive = np.repeat(additive[:, np.newaxis, :], seq_length, axis=1)
        data += additive

        mult = np.random.uniform(1 - freq_vary, 1 + freq_vary, [num_seqs, num_feats])
        mult = np.repeat(mult[:, np.newaxis, :], seq_length, axis=1)

        data *= mult

        split_pt = 0 # num_feats // 2
        data[:, :, split_pt:] = np.sin(data[:, :, split_pt:] * (2 * np.pi) / seq_period)
        data[:, :, split_pt:] = (data[:, :, split_pt:] + 1.) / 2.
        # data[:, :, :split_pt] = np.mod(data[:, :, :split_pt], seq_period) / seq_period

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

if __name__ == '__main__':

    from torch.utils.data import DataLoader

    dset = SequenceCopyDataset(
        num_feats=4,
        num_seqs=1200,
        seq_length=100,
        seq_period=12,
        freq_vary=0.6)

    dloader = DataLoader(dset, 10)

    for batch in dloader:
        print(batch.shape)
