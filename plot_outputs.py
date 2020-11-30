import numpy as np
import torch
import matplotlib.pyplot as plt
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


def plot_notetuple(inp, output, target):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    pr = np.zeros([inp.shape[0], 129])
    # Iterate over note names, which will be converted to note number later
    for i, n in enumerate(inp):
        pr[i, int(n[2])] = 1
        pr[i, -1] = int(n[1] == 0)

    axs[0].imshow(pr.T, aspect='auto', interpolation=None)
    axs[0].set_title('Input (with errors)')

    trg = target.cpu().detach().numpy()
    trg = np.stack([trg for _ in range(50)], 1)
    opt = torch.sigmoid(output).squeeze(-1).cpu().detach().numpy()
    opt = np.stack([opt for _ in range(50)], 1)

    locs = np.concatenate([trg, opt], 1)
    axs[1].imshow(locs.T, aspect='auto', interpolation=None)
    axs[1].set_title('Error locations + Predicted error locations')

    return fig, axs

# pitch_data = data[:, :, :-num_dur_vals]
# dur_data = data[:, :, -num_dur_vals:]
# plot(pitch_output, dur_output, pitch_data, dur_data,  7)
