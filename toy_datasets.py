from torch.utils.data import Dataset
import torch
import numpy as np


class SequenceCopyDataset(Dataset):

    def __init__(self, num_feats, num_seqs, seq_length, seq_period, phase_vary=0, freq_vary=0, sine=False):

        super(SequenceCopyDataset).__init__()
        self.seq_length = seq_length
        self.seq_period = seq_period
        self.num_feats = num_feats
        self.num_seqs = num_seqs

        data = np.zeros([num_seqs, seq_length, num_feats])
        slope = np.linspace(1, seq_length, seq_length)
        data[:] = np.tile(slope, [num_feats, 1]).swapaxes(0, 1)

        additive = np.random.normal(0, phase_vary, [num_seqs, num_feats])
        additive = np.repeat(additive[:, np.newaxis, :], seq_length, axis=1)
        data += additive

        mult = np.random.uniform(-freq_vary, freq_vary, [num_seqs, num_feats]) + 1
        mult = np.repeat(mult[:, np.newaxis, :], seq_length, axis=1)
        data *= mult

        if sine:
            data = np.sin(data * (2 * np.pi) / seq_period)
        # data[:, :, split_pt:] = (data[:, :, split_pt:] + 1.) / 2.
        else:
            data = np.mod(data, seq_period) / seq_period

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dset = SequenceCopyDataset(
        num_feats=1,
        num_seqs=1200,
        seq_length=128,
        seq_period=30,
        phase_vary=0.01,
        freq_vary=0.01)

    dloader = DataLoader(dset, 10)

    for batch in dloader:
        print(batch.shape)

    x = batch.numpy()
    for i in range(4):
        plt.plot(x[i], label="line {i}")
    plt.show()
