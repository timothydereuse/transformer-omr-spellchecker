import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
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


def plot_notetuple(inp, output, target, thresh=None):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    inp = inp.cpu().detach().numpy()

    pr = np.zeros([inp.shape[0], 129])
    # Iterate over note names, which will be converted to note number later
    for i, n in enumerate(inp):
        pitch = np.clip(n[2], 0, 127)
        pr[i, int(pitch)] = 1
        pr[i, -1] = int(n[1] == 0)

    axs[0].imshow(pr.T, aspect='auto', interpolation=None)
    axs[0].set_title('Input (with errors)')

    trg = target.cpu().detach().numpy()
    opt = output.squeeze(-1).cpu().detach().numpy()

    if thresh:
        opt = opt > thresh

    locs = np.concatenate([trg, opt], 1)
    # repeat entries a bunch of times because matplotlib doesn't
    # respect the interpolation=None entry sometimes, which looks ugly
    locs = np.array([locs[:, x] for x in np.repeat(range(locs.shape[1]), 25)])
    axs[1].imshow(locs, aspect='auto', interpolation=None)
    axs[1].set_title('Error locations + Predicted error locations')

    return fig, axs


def plot_pianoroll_corrections(orig, err, tgt_corr, pred_corr, thresh):
    orig = orig.cpu().numpy().round().astype('int')
    err = err.cpu().numpy().round().astype('int')
    tgt_corr = tgt_corr.cpu().numpy().astype('int')
    pred_corr = (pred_corr.cpu().numpy() > thresh).astype('int')

    pr_length = max(orig[:, 1].sum(), err[:, 1].sum())

    def make_pr(notes, corr=None):
        pr = np.zeros([pr_length, 128])
        cur_time = 0
        for i, e in enumerate(notes):
            note_val = 1
            if e[2] > 127:
                continue
            elif corr is None:
                pass
            elif corr[i, 0]:
                note_val = 2
            elif corr[i, 1]:
                note_val = 3
            elif corr[i, 2]:
                note_val = 4
            pr[cur_time:cur_time + e[0], e[2]] = note_val
            cur_time += e[1]

        pr = pr[:, np.any(pr > 0, axis=0)]

        return pr

    original_pr = make_pr(orig)
    real_corrected = make_pr(err, tgt_corr)
    pred_corrected = make_pr(err, pred_corr)

    cmap = colors.ListedColormap(['black', 'gray', 'yellow', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs[0].imshow(original_pr.T, aspect='auto', interpolation='Nearest', cmap=cmap, norm=norm)
    axs[0].set_title('Real Music')

    axs[1].imshow(real_corrected.T, aspect='auto', interpolation='Nearest', cmap=cmap, norm=norm)
    axs[1].set_title('Musical Input (real errors)')

    axs[2].imshow(pred_corrected.T, aspect='auto', interpolation='Nearest', cmap=cmap, norm=norm)
    axs[2].set_title('Musical Input (predicted errors)')

    # fig.savefig('test.png', bbox_inches='tight')

    return fig, axs


def plot_set(exs, dset, ind=0):
    target = dset.unnormalize_batch(exs['target']).detach().numpy().astype(int)
    output = dset.unnormalize_batch(exs['output']).detach().numpy().astype(int)
    inp = dset.unnormalize_batch(exs['input']).detach().numpy().astype(int)

    fig, axs = plt.subplots(3, 1, figsize=(9, 6))
    max_pitch = np.max(inp[ind, :, 1])
    max_time = np.max(inp[ind, :, 0])

    for i, p in enumerate([inp, target, output]):
        axs[i].set_title(('input', 'target', 'output')[i])
        axs[i].scatter(p[ind, :, 0], p[ind, :, 1], s=p[ind, :, 2] / 2, c=p[ind, :, 3])
        axs[i].set_xlim(0, max_time)
        axs[i].set_ylim(0, max_pitch)

    return fig, axs

# pitch_data = data[:, :, :-num_dur_vals]
# dur_data = data[:, :, -num_dur_vals:]
# plot(pitch_output, dur_output, pitch_data, dur_data,  7)
