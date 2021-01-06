# contains functions that take in single batches of symbolic music as multidimensional tensors and
# return them as (input, target) pairs. this involves removing or otherwise degrading the input
# in some way. the intent is that these are used "live," in the training loop itself.

from torch.utils.data import DataLoader
import torch
import numpy as np
import model_params as params
from difflib import SequenceMatcher


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
    batch_size = inp.shape[0]

    # make list of indices to mask / corrupt; select exactly @num_indices from each sequence

    inds_selected = []
    inds_left = []

    for i in range(batch_size):
        inds = np.arange(seq_len)
        np.random.shuffle(inds)
        inds_selected += list(zip([i] * num_indices, inds[:num_indices]))
        inds_left += list(zip([i] * (seq_len - num_indices), inds[num_indices:]))

    inds_selected = np.array(inds_selected)
    inds_left = np.array(inds_left)

    np.random.shuffle(inds_selected)
    num_rand = int(prob_random * len(inds_selected))
    num_same = int(prob_same * len(inds_selected))

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
    mask_element = torch.tensor(mask_element).to(output.device).float()
    output[inds_mask[:, 0], inds_mask[:, 1]] = mask_element

    return output, (inds_mask, inds_rand, inds_left)


def error_indices(inp, num_indices=5):
    '''
    adding errors systematically to batches of notetuple-format data. should happen on cpu only.
    '''

    seq_len = inp.shape[1]
    batch_size = inp.shape[0]
    pad_seq = torch.tensor([params.notetuple_flags['pad'] for _ in range(seq_len)], dtype=inp.dtype)
    output = inp.clone()

    means = inp.float().view(-1, 3).mean(0)
    stds = inp.float().view(-1, 3).std(0)
    errored_indices = np.zeros((batch_size, seq_len, 3))

    for i in range(batch_size):
        entry = output[i]

        inds = np.arange(seq_len)
        np.random.shuffle(inds)
        sel_inds = inds[:num_indices]
        # errored_indices[i, sel_inds] = 1

        # make errors from distribution of actual data
        errors = torch.normal(0.0, 1.0, (num_indices, 3)) * stds + means
        entry[sel_inds] = errors.round().abs()

        # delete masked entries
        mask = np.ones(len(entry), dtype='bool')
        inds_delete = inds[num_indices:2*num_indices]
        mask[inds_delete] = False
        entry = entry[mask]

        # more errors to insert instead of replace
        inds_insert = inds[2*num_indices:3*num_indices] % len(entry)
        errors = torch.normal(0.0, 1.0, (num_indices, 3)) * stds + means
        for n in range(num_indices):
            entry = np.insert(entry, inds_insert[n], errors[n], 0)

        errored_indices[i] = get_notetuple_diff(entry, inp[i])

        # add padding in case overall sequence length has changed, then cut down
        # to length of original output
        end = min(len(entry), output.shape[0])
        entry = torch.cat([entry, pad_seq], 0)
        output[i, :end] = entry[:end]

    return output, torch.tensor(errored_indices, dtype=inp.dtype)


def get_notetuple_diff(err, orig):
    err = [tuple(x) for x in err.numpy()]
    orig = [tuple(x) for x in orig.numpy()]

    s = SequenceMatcher(None, err, orig)
    ops = [x for x in s.get_opcodes() if not x[0] == 'equal']

    # replace / insert / delete
    record = np.zeros([len(orig), 3])
    mapping = {
        'replace': 0,
        'insert': 1,
        'delete': 2
    }

    for item in ops:
        end_index = max(item[2], item[1] + 1)
        type_ind = mapping[item[0]]
        record[item[1]:end_index, type_ind] = 1

    return record


if __name__ == '__main__':
    import data_loaders as dl
    dset = dl.MidiNoteTupleDataset(
        dset_fname=params.dset_path,
        seq_length=params.seq_length,
        base='train',
        padding_amt=params.padding_amt,
        trial_run=params.trial_run)

    dload = DataLoader(dset, batch_size=40)
    for i, batch in enumerate(dload):
        batch = batch.float()
        print(i, batch.shape)
        if i > 2:
            break

    kw = {'num_indices': 5}
    errored, indices = error_indices(batch, **kw)


    # model = tfsm.TransformerModel(
    #     num_feats=dset.num_feats,
    #     nlayers=1
    # )
    # model = model.float()

    # out = model(inp, tgt)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(inp.numpy()[:, 0].T)
    # plt.show()
