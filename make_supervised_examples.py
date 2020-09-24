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

    output = input.clone()
    output[:, inds_to_remove] = torch.zeros_like(input[:, inds_to_remove])

    return output, target


def mask_indices(inp, num_indices=1, prob_random=0.15, prob_same=0.15, continguous=False):
    '''
    masked language model procedure used in original BERT paper to train bidirectional tranformer.
    '''
    seq_len = inp.shape[1]

    # make list of indices to mask / corrupt.
    inds_selected = np.array([
        list(zip(
            [x] * num_indices,
            np.random.choice(seq_len, num_indices, False)
        ))
        for x
        in range(inp.shape[0])
    ]).reshape(-1, 2)

    output = inp.clone()

    flattened_inp = inp.reshape(-1, inp.shape[-1])
    num_els = flattened_inp.shape[0]

    for ind in inds_selected:
        r = np.random.rand()
        if r < prob_random:
            loc = np.random.randint(num_els)
            slice = flattened_inp[loc].clone()
            output[tuple(ind)] = slice
        elif r > (1 - prob_random):
            pass
        else:
            output[tuple(ind)] = torch.zeros_like(output[tuple(ind)])

    return output, inds_selected


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
