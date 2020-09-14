# contains functions that take in single batches of symbolic music as multidimensional tensors and
# return them as (input, target) pairs. this involves removing or otherwise degrading the input
# in some way. the intent is that these are used "live," in the training loop itself.

from torch.utils.data import IterableDataset, DataLoader
import factorizations as fcts
from importlib import reload
import torch
import numpy as np


def remove_indices(input, num_indices=1, mode='center'):

    if mode not in ['center', 'start', 'end', 'batch_random', 'entry_random']:
        raise ValueError(f'mode {mode} invalid for remove_indices')

    target_length = 3
    target_position = 'center'

    middle_ind = batch.shape[1] / 2
    st = int(middle_ind - (target_length / 2))
    end = int(middle_ind + (target_length / 2))
    inds_to_remove = list(range(st, end))

    target = batch.clone()[:, inds_to_remove]
    batch[:, inds_to_remove] = 1

    return batch, target


if __name__ == '__main__':
    import data_loaders as dl
    fname = 'essen_meertens_songs.hdf5'
    num_dur_vals = 17
    seq_len = 30
    proportion = 0.2
    dset = dl.MonoFolkSongDataset(fname, seq_len, num_dur_vals=num_dur_vals,
                                  proportion_for_stats=proportion)

    dload = DataLoader(dset, batch_size=20)
    for i, batch in enumerate(dload):
        batch = batch.float()
        print(i, batch.shape)
        if i > 2:
            break

    inp, tgt = remove_indices(batch)
    inp = inp.transpose(1, 0)
    tgt = tgt.transpose(1, 0)

    import transformer_full_seq_model as tfsm

    model = tfsm.TransformerModel(
        num_feats=dset.num_feats,
        nlayers=1
    )
    model = model.float()

    out = model(inp, tgt)
