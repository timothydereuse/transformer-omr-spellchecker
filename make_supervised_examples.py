# contains functions that take in single batches of symbolic music as multidimensional tensors and
# return them as (input, target) pairs. this involves removing or otherwise degrading the input
# in some way. the intent is that these are used "live," in the training loop itself.

from torch.utils.data import DataLoader
import torch
import numpy as np
import model_params
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


def error_indices(inp, num_deletions=5, num_insertions=5, num_replacements=5):
    '''
    adding errors systematically to batches of notetuple-format or point-set format data.
    should happen on cpu only.
    '''

    seq_len = inp.shape[1]
    batch_size = inp.shape[0]
    num_feats = inp.shape[-1]
    pad_element = model_params.notetuple_flags['pad'][:num_feats]
    pad_seq = np.array([pad_element for _ in range(seq_len)])
    output = inp.clone().numpy()

    means = inp.float().view(-1, num_feats).mean(0).numpy()
    stds = inp.float().view(-1, num_feats).std(0).numpy()

    # max_num_error_pts = num_deletions + num_insertions + (2 * num_replacements)

    errored_indices = np.zeros([batch_size, seq_len, 1])

    for i in range(batch_size):
        entry = output[i]
        # max_time = np.max(entry[:, 0])

        # replacements:
        inds = np.arange(seq_len)
        np.random.shuffle(inds)
        sel_inds = inds[:num_replacements]
        errors = np.random.normal(0.0, 1.0, (num_replacements, num_feats)) * (stds / 3)
        errors = (np.round(errors))
        entry[sel_inds] = entry[sel_inds] + errors

        # deletions:
        mask = np.ones(len(entry), dtype='bool')
        inds_delete = inds[num_replacements:num_replacements + num_deletions]
        mask[inds_delete] = False
        entry = entry[mask]

        # more errors to insert instead of replace
        inds_insert = inds[-num_insertions:] % len(entry)
        errors = np.random.normal(0.0, 1.0, (num_insertions, num_feats)) * stds + means
        errors = np.abs(np.round(errors))
        for n in range(num_insertions):
            entry = np.insert(entry, inds_insert[n], errors[n], 0)

        # # wrap around anything too far forward in time
        # entry[:, 0] = entry[:, 0] % max_time

        # set_xor = error_set_xor(entry, inp[i].numpy())
        # set_xor = np.concatenate([set_xor, pad_seq], 0)
        # errored_indices[i] = set_xor[:max_num_error_pts]

        diff = get_notetuple_diff(entry, inp[i].numpy())

        # add padding in case overall sequence length has changed, then cut down
        # to length of original output
        entry = np.concatenate([entry, pad_seq], 0)
        output[i, :seq_len] = entry[:seq_len]

        errored_indices[i] = diff

    output = torch.tensor(output, dtype=inp.dtype)
    errored_indices = torch.tensor(errored_indices, dtype=inp.dtype)

    return output, errored_indices


def get_notetuple_diff(err, orig, for_autoregressive=False):
    err = [tuple(x) for x in err]
    orig = [tuple(x) for x in orig]
    print(len(err), len(orig))

    s = SequenceMatcher(None, err, orig, autojunk=False)
    ops = [x for x in s.get_opcodes() if not x[0] == 'equal']
    # ops = s.get_opcodes()

    # replace / insert / delete
    mapping = {
        'replace': 0,
        'insert': 1,
        'delete': 2
    }

    print(ops)

    if not for_autoregressive:
        record = np.zeros([len(orig), 1])
        for item in ops:
            end_index = max(item[2], item[1] + 1)
            record[item[1]:end_index, 0] = 1
    else:
        record = []
        for item in ops:
            start_index = item[1]
            amt = (item[2] - item[1]) if (item[2] - item[1]) > 0 else item[4] - item[3]
            type_ind = mapping[item[0]]
            record.append([type_ind, start_index, amt])
        record = np.stack(record, 0)
    return record


def autoregressive_target(batch, teacher_forcing=True, num_indices=10):
    if teacher_forcing:
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
    else:
        inp = batch[:, :-num_indices]
        tgt = batch[:, -num_indices:]

    return inp, tgt


def error_set_xor(err, orig):
    err_set = set(map(tuple, err.astype('int')))
    orig_set = set(map(tuple, orig.astype('int')))
    x = err_set.symmetric_difference(orig_set)
    x = np.array(list(x))
    return x


if __name__ == '__main__':
    import point_set_dataloader as dl

    params = model_params.Params()

    dset = dl.MidiNoteTupleDataset(
        dset_fname=params.dset_path,
        seq_length=params.seq_length,
        num_feats=params.num_feats,
        base='train',
        padding_amt=params.padding_amt)

    dload = DataLoader(dset, batch_size=2)
    for i, batch in enumerate(dload):
        batch = batch.float()
        print(i, batch.shape)
        if i > 2:
            break

    kw = {'num_insertions': 3, 'num_deletions': 3, 'num_replacements': 3}
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
