import numpy as np
import torch
import matplotlib.pyplot as plt


def get_pr(inp, ind=0):
    slice = torch.transpose(inp, 1, 0)[ind]
    maxinds = slice.max(1).indices
    for i, x in enumerate(maxinds):
        slice[i] = 0
        slice[i][x] += 1
    pr = slice.cpu().detach().numpy()
    return pr


def plot(outputs, targets, ind, num_dur_vals, errored=None):
    if errored is None:
        fig, axs = plt.subplots(1, 3, figsize=(6, 4))
    else:
        fig, axs = plt.subplots(1, 4, figsize=(8, 4))

    full_out = outputs[:, ind].cpu().detach().numpy()
    axs[0].imshow(full_out.T)
    axs[0].set_title('Raw Transformer \n Reconstruction')

    pitch_out = outputs[:, :, :-num_dur_vals]
    dur_out = outputs[:, :, -num_dur_vals:]
    do_pr = get_pr(dur_out, ind)
    po_pr = get_pr(pitch_out, ind)
    pro = np.concatenate([po_pr, do_pr], 1)
    axs[1].imshow(pro.T)
    axs[1].set_title('Thresholded \n Reconstruction')

    targets = targets[:, ind].cpu().detach().numpy()
    axs[2].imshow(targets.T)
    axs[2].set_title('Ground Truth')

    if errored is not None:
        pitch_out = errored[:, :, :-num_dur_vals]
        dur_out = errored[:, :, -num_dur_vals:]
        do_pr = torch.transpose(dur_out, 1, 0)[ind].cpu().detach().numpy()
        po_pr = torch.transpose(pitch_out, 1, 0)[ind].cpu().detach().numpy()
        pro = np.concatenate([po_pr, do_pr], 1)
        axs[3].imshow(pro.T)
        axs[3].set_title('Input \n (with errors)')


    return fig, axs


# pitch_data = data[:, :, :-num_dur_vals]
# dur_data = data[:, :, -num_dur_vals:]
# plot(pitch_output, dur_output, pitch_data, dur_data,  7)
