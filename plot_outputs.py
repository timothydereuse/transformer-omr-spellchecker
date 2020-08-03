import numpy as np
import torch
import matplotlib.pyplot as plt
from importlib import reload


def get_pr(inp, ind=0):
    slice = torch.transpose(inp, 1, 0)[ind]
    maxinds = slice.max(1).indices
    for i, x in enumerate(maxinds):
        slice[i] = torch.sigmoid(slice[i])
        slice[i][x] += 1
    pr = slice.cpu().detach().numpy()
    return pr


def plot(pitch_out, dur_out, pitch_in, dur_in, ind):
    fig, axs = plt.subplots(1, 2, figsize=(4, 8))

    do_pr = get_pr(dur_out, ind)
    po_pr = get_pr(pitch_out, ind)
    pro = np.concatenate([po_pr, do_pr], 1)
    axs[0].imshow(pro.T)

    di_pr = get_pr(dur_in, ind)
    pi_pr = get_pr(pitch_in, ind)
    pri = np.concatenate([pi_pr, di_pr], 1)
    axs[1].imshow(pri.T)

    fig.show()


# pitch_data = data[:, :, :-num_dur_vals]
# dur_data = data[:, :, -num_dur_vals:]
# plot(pitch_output, dur_output, pitch_data, dur_data,  7)
