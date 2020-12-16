import torch
import data_loaders as dl
import plot_outputs as po
import numpy as np
import make_supervised_examples as mse
from torch.utils.data import IterableDataset, DataLoader
import model_params as params
from importlib import reload

reload(params)
reload(mse)


def f_measure(inps, targets, threshold=0.5):
    thresh_predictions = (inps > threshold)
    tru_pos = np.logical_and(thresh_predictions, targets).sum()
    pred_pos = thresh_predictions.sum()
    targ_pos = targets.sum()

    if pred_pos > 0 and targ_pos > 0 and tru_pos > 0:
        precision = tru_pos / pred_pos
        recall = tru_pos / targ_pos
        F1 = (precision * recall * 2) / (precision + recall)
    else:
        F1 = 0

    return F1


def multilabel_thresholding(output, target, num_trials=1000):
    output = output.cpu().detach().numpy().reshape(-1)
    target = target.cpu().detach().numpy().reshape(-1)

    # sort_output = np.sort(output)
    # thresholds = np.interp(
    #     x=np.linspace(0, len(sort_output), num_trials),
    #     xp=np.arange(len(sort_output)),
    #     fp=sort_output
    # )

    thresholds = np.linspace(min(output), max(output), num_trials)

    F1s = np.zeros(output.shape)
    for i, t in enumerate(thresholds):
        F1s[i] = f_measure(output, target, t)

    best_thresh = thresholds[np.argmax(F1s)]
    best_score = np.max(F1s)

    return best_score, best_thresh
