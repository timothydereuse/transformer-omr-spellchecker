from torch.utils.data import Dataset
import torch
import numpy as np


class SequenceCopyDataset(Dataset):
    """meertens_tune_collection dataset."""

    def __init__(self, num_feats, num_seqs, seq_length, seq_period, seq_type='sine'):
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
        data[:] = np.tile(slope, [4, 1]).swapaxes(0, 1)

        additive = np.random.normal(0, 10, [num_seqs, num_feats])
        data += np.repeat(additive[:, np.newaxis, :], seq_length, axis=1)

        mult = np.random.uniform(0.75, 1.25, [num_seqs])
        mult = np.repeat(mult[:, np.newaxis], seq_length, axis=1)
        mult = np.repeat(mult[:, :, np.newaxis], num_feats, axis=2)
        data *= mult

        data = np.sin(data / (2 * np.pi))

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

if __name__ == '__main__':

    from torch.utils.data import DataLoader

    dset = SequenceCopyDataset(4, 1000, 50, 'sine', 14)

    dloader = DataLoader(dset, 10)

    for batch in dloader:
        print(batch.shape)
