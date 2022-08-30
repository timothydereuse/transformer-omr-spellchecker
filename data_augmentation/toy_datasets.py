from torch.utils.data import Dataset
import torch
import numpy as np


class SequenceCopyDataset(Dataset):

    def __init__(self, num_feats, num_seqs, seq_length, seq_freq, phase_vary=0, freq_vary=1, power_vary=1, seq_type='sine', rand_freq=None):

        if type(seq_type) == list and len(seq_type) != num_feats:
            raise ValueError("List of sequence types must have same length as number of features")
        if type(seq_type) == str:
            seq_type = [seq_type for _ in range(num_feats)]
        if not rand_freq:
            rand_freq = int((seq_length / seq_freq) // 4)

        super(SequenceCopyDataset).__init__()
        self.seq_length = seq_length
        self.seq_freq = seq_freq
        self.seq_period = seq_length / seq_freq
        self.num_feats = num_feats
        self.num_seqs = num_seqs

        data = np.zeros([num_seqs, seq_length, num_feats])
        slope = np.linspace(1, seq_length, seq_length)
        data[:] = np.tile(slope, [num_feats, 1]).swapaxes(0, 1)

        add_amt = phase_vary * seq_length
        additive = np.random.uniform(0, add_amt, [num_seqs, num_feats])
        additive = np.repeat(additive[:, np.newaxis, :], seq_length, axis=1)
        data += additive

        logv = np.log(freq_vary)
        mult = np.random.uniform(-logv, logv, [num_seqs])
        mult = np.exp(mult)
        mult = np.repeat(mult[:, np.newaxis], seq_length, axis=1)
        mult = np.repeat(mult[:, :, np.newaxis], num_feats, axis=2)
        data *= mult

        for i in range(num_feats):
            if seq_type[i] == 'saw':
                data[:, :, i] = np.mod(data[:, :, i], self.seq_period) / self.seq_period
            elif seq_type[i] == 'sine':
                data[:, :, i] = np.sin(data[:, :, i] * (2 * np.pi) / self.seq_period) * 0.5 + 0.5
            elif seq_type[i] == 'rand':
                data[:, :, i] = np.mod(data[:, :, i], self.seq_period) / self.seq_period
                x = np.linspace(0, 1, rand_freq)
                for s in range(self.num_seqs):
                    y = np.linspace(0, 1, rand_freq)
                    np.random.shuffle(y)
                    data[s, :, i] = np.interp(data[s, :, i], x, y)

        # apply power
        logp = np.log(power_vary)
        power = np.random.uniform(-logp, logp, [num_seqs, num_feats])
        power = np.exp(power)
        power = np.repeat(power[:, np.newaxis, :], seq_length, axis=1)
        data = data ** power

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    num_feats = 3

    dset = SequenceCopyDataset(
        num_feats=num_feats,
        num_seqs=1200,
        seq_length=256,
        seq_freq=8,
        phase_vary=1,
        freq_vary=1,
        power_vary=1,
        seq_type='rand')

    dloader = DataLoader(dset, 40)

    for batch in dloader:
        print(batch.shape)

    x = batch.numpy()
    for i in range(num_feats):
        plt.plot(x[0, :, i], label="line {i}")
    plt.show()
