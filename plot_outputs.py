import numpy as np
import torch
import matplotlib.pyplot as plt


def get_pr(inp, ind=0):
    slice = torch.transpose(inp, 1, 0)[ind]
    maxinds = slice.max(1).indices
    for i, x in enumerate(maxinds):
        slice[i] = torch.sigmoid(slice[i])
        slice[i][x] += 1
    pr = slice.cpu().detach().numpy()
    return pr


def plot(outputs, targets, ind, num_dur_vals):
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    full_out = outputs[:, ind].cpu().detach().numpy()
    axs[0].imshow(full_out.T)

    pitch_out = outputs[:, :, :-num_dur_vals]
    dur_out = outputs[:, :, -num_dur_vals:]
    do_pr = get_pr(dur_out, ind)
    po_pr = get_pr(pitch_out, ind)
    pro = np.concatenate([po_pr, do_pr], 1)
    axs[1].imshow(pro.T)

    targets = targets[:, ind].cpu().detach().numpy()
    axs[2].imshow(targets.T)

    return fig, axs


# pitch_data = data[:, :, :-num_dur_vals]
# dur_data = data[:, :, -num_dur_vals:]
# plot(pitch_output, dur_output, pitch_data, dur_data,  7)
