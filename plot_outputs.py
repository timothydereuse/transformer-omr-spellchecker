import numpy as np
import torch
# import matplotlib.pyplot as plt
# from matplotlib import colors
import model_params as params


def get_pr(inp, ind=0):
    slice = inp[ind]
    maxinds = slice.max(1).indices
    for i, x in enumerate(maxinds):
        slice[i] = 0
        slice[i][x] += 1
    pr = slice.cpu().detach().numpy()
    return pr


def plot(outputs, targets, ind, num_dur_vals, errored=None):
    if errored is None:
        fig, axs = plt.subplots(1, 3, figsize=(9, 6))
    else:
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))

    full_out = outputs[ind, :].cpu().detach().numpy()
    axs[0].imshow(full_out.T)
    axs[0].set_title('Raw Transformer \n Reconstruction')

    if num_dur_vals > 0:
        pitch_out = outputs[:, :, :-num_dur_vals]
        dur_out = outputs[:, :, -num_dur_vals:]
        do_pr = get_pr(dur_out, ind)
        po_pr = get_pr(pitch_out, ind)
        pro = np.concatenate([po_pr, do_pr], 1)
    else:
        pro = get_pr(outputs, ind)

    axs[1].imshow(pro.T)
    axs[1].set_title('Thresholded \n Reconstruction')

    targets = targets[ind, :].cpu().detach().numpy()
    axs[2].imshow(targets.T)
    axs[2].set_title('Ground Truth')

    if errored is not None:
        if num_dur_vals > 0:
            pitch_out = errored[:, :, :-num_dur_vals]
            dur_out = errored[:, :, -num_dur_vals:]
            do_pr = dur_out[ind].cpu().detach().numpy()
            po_pr = pitch_out[ind].cpu().detach().numpy()
            pro = np.concatenate([po_pr, do_pr], 1)
        else:
            pro = get_pr(errored, ind)
        axs[3].imshow(pro.T)
        axs[3].set_title('Input \n (with errors)')

    return fig, axs


def plot_line_corrections(inp, output, target, thresh=None):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    inp = inp.cpu().detach().numpy()

    # Iterate over number of features
    for i in range(inp.shape[1]):
        axs[0].plot(inp[:, i])
    axs[0].set_title('Input (with errors)')

    trg = target.cpu().detach().numpy()

    if thresh:
        output = output > thresh
    else:
        output = output  # torch.sigmoid(output)
    opt = output.cpu().detach().numpy()

    locs = np.concatenate([trg, opt], 1)
    # repeat entries a bunch of times because matplotlib doesn't
    # respect the interpolation=None entry sometimes, which looks ugly
    locs = np.array([locs[:, x] for x in np.repeat(range(locs.shape[1]), 25)])
    axs[1].imshow(locs, aspect='auto', interpolation=None)
    axs[1].set_title('Error locations + Predicted error locations')

    return fig, axs


def plot_agnostic_results(exs, vocabulary, thresh, ind=-1, return_arrays=False):

    if ind < 0:
        ind = np.random.choice(exs['input'].shape[0])
    target = exs['target'][ind].detach().cpu().numpy().astype('int')
    output = (exs['output'][ind].detach().cpu().numpy() > thresh).astype('int')
    orig = exs['orig'][ind].detach().cpu().numpy().astype('int')
    input = exs['input'][ind].detach().cpu().numpy().astype('int')

    input_words = vocabulary.vec_to_words(input)
    orig_words = vocabulary.vec_to_words(orig)
    space = ' '
    mark = 'X'

    if return_arrays:
        res = []
        for i in range(len(input)):
            res.append([orig_words[i], input_words[i], target[i], output[i]])
        return res

    lines = ['ORIG | ERRORED INPUT | TARGET | OUTPUT \n']
    for i in range(len(input)):
        line = (
            f'{orig_words[i]:25} | {input_words[i]:25} '
            f' | {mark if target[i] else space}'
            f' | {mark if output[i] else space} | \n'
            )
        lines.append(line)

    return lines


def plot_pianoroll_corrections(exs, dset, thresh, ind=-1):

    # currently VOICE, ONSET, DURATION, PITCH
    INDVOICE = 0
    INDONSET = 1
    INDDUR = 2
    INDPITCH = 3
    
    if ind < 0:
        ind = np.random.choice(exs['input'].shape[0])
    target = exs['target'][ind].detach().cpu().numpy().astype('int')
    output = (exs['output'][ind].detach().cpu().numpy() > thresh).astype('int')
    # input = dset.unnormalize_batch(exs['input'][ind].detach().cpu()).numpy().astype('int')
    # orig = dset.unnormalize_batch(exs['orig'][ind].detach().cpu()).numpy().astype('int')
    orig = exs['orig'][ind].detach().cpu().numpy().astype('int')
    input = exs['input'][ind].detach().cpu().numpy().astype('int')

    pr_length = max(orig[:, INDONSET].sum(), input[:, INDONSET].sum())
    pr_length += int(max(orig[:, INDDUR]) + max(input[:, INDDUR]))

    def make_pr(notes, corr=None):
        pr = np.zeros([pr_length, 128])
        cur_time = 0
        for i, e in enumerate(notes):
            note_val = 1
            if e[INDPITCH] > 127:
                continue
            elif corr is None:
                pass
            elif corr[i]:
                note_val = 3
            pr[cur_time:cur_time + e[INDDUR], e[INDPITCH]] = note_val
            pr[cur_time, e[INDPITCH]] = note_val + 1
            cur_time += e[INDONSET]

        # pr = pr[:, np.any(pr > 0, axis=0)]
        used_notes = np.sum(pr, axis=0).nonzero()[0]
        lower, upper = (int(used_notes[0]), int(used_notes[-1]))
        pr = pr[:, lower:upper+1]
        # pr = np.flip(pr, 1) # flip so that rests (0) are on the bottom

        return pr

    original_pr = make_pr(orig)
    real_corrected = make_pr(input, target)
    pred_corrected = make_pr(input, output)

    cmap = colors.ListedColormap(['black', 'gray', 'white', 'orange', 'yellow'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs[0].imshow(original_pr.T, aspect='auto', interpolation='Nearest', cmap=cmap, norm=norm)
    axs[0].invert_yaxis()
    axs[0].set_title('Real Music')

    axs[1].imshow(real_corrected.T, aspect='auto', interpolation='Nearest', cmap=cmap, norm=norm)
    axs[1].invert_yaxis()
    axs[1].set_title('Musical Input (real errors)')

    axs[2].imshow(pred_corrected.T, aspect='auto', interpolation='Nearest', cmap=cmap, norm=norm)
    axs[2].invert_yaxis()
    axs[2].set_title('Musical Input (predicted errors)')

    # fig.savefig('test.png', bbox_inches='tight')

    return fig, axs


def plot_set(exs, dset, ind=0):
    target = dset.unnormalize_batch(exs['target'].detach().cpu()).numpy().astype(int)
    output = dset.unnormalize_batch(exs['output'].detach().cpu()).numpy().astype(int)
    inp = dset.unnormalize_batch(exs['input'].detach().cpu()).numpy().astype(int)
    num_feats = inp.shape[-1]

    fig, axs = plt.subplots(3, 1, figsize=(9, 6))
    max_pitch = np.max(inp[ind, :, 1])
    max_time = np.max(inp[ind, :, 0])

    for i, p in enumerate([inp, target, output]):
        axs[i].set_title(('input', 'target', 'output')[i])
        if num_feats == 2:
            s, c = (6, 'black')
        elif num_feats == 3:
            s, c = (6, p[ind, :, 2])
        else:
            s, c = (p[ind, :, 3] / 2, p[ind, :, 2])
        axs[i].scatter(p[ind, :, 0], p[ind, :, 1], s=s, c=c)
        axs[i].set_xlim(-5, max_time + 5)
        axs[i].set_ylim(-5, max_pitch + 5)

    return fig, axs

# pitch_data = data[:, :, :-num_dur_vals]
# dur_data = data[:, :, -num_dur_vals:]
# plot(pitch_output, dur_output, pitch_data, dur_data,  7)
