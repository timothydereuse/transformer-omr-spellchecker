# contains functions that take in single batches of symbolic music as multidimensional tensors and
# return them as (input, target) pairs. this involves removing or otherwise degrading the input
# in some way. the intent is that these are used "live," in the training loop itself.

from torch.utils.data import DataLoader
import torch
import numpy as np


def remove_indices(input, num_indices=1, mode='center'):

    if mode == 'center':
        middle_ind = input.shape[1] / 2
        st = int(middle_ind - (num_indices / 2))
        end = int(middle_ind + (num_indices / 2))
        inds_to_remove = list(range(st, end))
    elif mode == 'batch_random':
        st = np.random.randint(input.shape[1] - num_indices)
        end = st + num_indices
        inds_to_remove = list(range(st, end))
    else:
        raise ValueError(f'mode {mode} invalid for remove_indices')

    target = input.clone()[:, inds_to_remove]

    # print(inds_to_remove, target.shape, input.shape)

    input[:, inds_to_remove] = torch.rand_like(input[:, inds_to_remove])

    return input, target


if __name__ == '__main__':
    import data_loaders as dl
    fname = 'essen_meertens_songs.hdf5'
    num_dur_vals = 17
    seq_len = 20
    proportion = 1
    dset = dl.MonoFolkSongDataset(fname, seq_len, num_dur_vals=num_dur_vals,
                                  proportion_for_stats=proportion)

    dload = DataLoader(dset, batch_size=1000)
    for i, batch in enumerate(dload):
        batch = batch.float()
        print(i, batch.shape)
        if i > 2:
            break

    kw = {'mode': 'center', 'num_indices': 2}
    inp, tgt = remove_indices(batch, **kw)
    inp = inp.transpose(1, 0)
    tgt = tgt.transpose(1, 0)

    import transformer_full_seq_model as tfsm

    # model = tfsm.TransformerModel(
    #     num_feats=dset.num_feats,
    #     nlayers=1
    # )
    # model = model.float()

    # out = model(inp, tgt)
    #
    import matplotlib.pyplot as plt
    plt.imshow(inp.numpy()[:, 0].T)
    plt.show()
