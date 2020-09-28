# contains functions that take in single batches of symbolic music as multidimensional tensors and
# return them as (input, target) pairs. this involves removing or otherwise degrading the input
# in some way. the intent is that these are used "live," in the training loop itself.

from torch.utils.data import DataLoader
import torch
import numpy as np
import model_params as params


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


def mask_indices(inp, num_indices=5, prob_random=0.15, prob_same=0.15, continguous=False):
    '''
    masked language model procedure used in original BERT paper to train bidirectional tranformer.
    '''
    seq_len = inp.shape[1]

    # make list of indices to mask / corrupt; select exactly @num_indices from each sequence
    inds_selected = np.array([
        list(zip(
            [x] * num_indices,
            np.random.choice(seq_len, num_indices, False)
        ))
        for x in range(inp.shape[0])
    ]).reshape(-1, 2)

    np.random.shuffle(inds_selected)
    num_rand = int(prob_random * inds_selected.shape[0])
    num_same = int(prob_same * inds_selected.shape[0])

    inds_rand = inds_selected[:num_rand]
    inds_mask = inds_selected[num_rand:-num_same]
    # discard indices in region of num_same

    output = inp.clone()

    # randomize the order of the batch, get a couple of them, and replace rand_ind of the
    # selected indices with other sequence elements in the batch
    flattened_inp = inp.reshape(-1, inp.shape[-1])
    replacer = flattened_inp[np.random.choice(flattened_inp.shape[0], inds_rand.shape[0])]
    output[inds_rand[:, 0], inds_rand[:, 1]] = replacer

    mask_element = np.zeros((len(inds_mask), inp.shape[-1]))
    mask_element[:, params.flags['mask']] = 1
    output[inds_mask[:, 0], inds_mask[:, 1]] = torch.tensor(mask_element, dtype=torch.float)

    return output, (inds_mask, inds_rand)


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

    kw = {'num_indices': 5, 'prob_same': 0}
    inp, _ = mask_indices(batch, **kw)
    inp = inp.transpose(1, 0)
    tgt = batch.transpose(1, 0)

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
