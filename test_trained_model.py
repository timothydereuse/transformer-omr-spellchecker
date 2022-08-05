import torch
import torch.nn as nn
import plot_outputs as po
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader
import models.LSTUT_model as lstut
import model_params as params
from importlib import reload
import data_management.vocabulary as vocab
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay


reload(params)
reload(po)


def test_results(output, target, results_dict, threshold):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    predictions = (output > threshold)

    for i, cat in enumerate(['replace', 'insert', 'delete']):
        cat_preds = predictions[:, :, i].reshape(-1).astype('bool')
        cat_target = target[:, :, i].reshape(-1).astype('bool')

        t_pos = (cat_preds & cat_target).sum()
        results_dict[cat]['t_pos'].append(t_pos)

        t_neg = (~cat_preds & ~cat_target).sum()
        results_dict[cat]['t_neg'].append(t_neg)

        f_pos = (cat_preds & ~cat_target).sum()
        results_dict[cat]['f_pos'].append(f_pos)

        f_neg = (~cat_preds & cat_target).sum()
        results_dict[cat]['f_neg'].append(f_neg)

    return results_dict

def precision_recall(inps, targets, threshold=0.5):
    thresh_predictions = (inps > threshold)
    tru_pos = np.logical_and(thresh_predictions, targets).sum()
    pred_pos = thresh_predictions.sum()
    targ_pos = targets.sum()

    precision = tru_pos / pred_pos if pred_pos > 0 else 0
    recall = tru_pos / targ_pos if targ_pos > 0 else 0

    return precision, recall

def find_thresh_for_given_recall(output, target, num_trials=10000, target_recall=0.5):
    if type(output) == torch.Tensor:
        output = output.cpu().detach().numpy()
    output = output.reshape(-1)

    if type(target) == torch.Tensor:
        target = target.cpu().detach().numpy()
    target = target.reshape(-1)

    thresholds = np.linspace(min(output), max(output), num_trials)

    recalls = np.zeros(thresholds.shape)
    for i, t in enumerate(thresholds):
        _, recalls[i] = precision_recall(output, target, threshold=t)

    closest_thresh = thresholds[np.argmin(np.abs(recalls - target_recall))]

    return closest_thresh

def find_thresh_for_given_recalls(output, target, target_recalls, num_trials=1000):
    return [
        find_thresh_for_given_recall(output, target, num_trials, x)
        for x
        in target_recalls
    ]

def f_measure(inps, targets, threshold=0.5, beta=1):
    beta_squared = beta * beta
    thresh_predictions = (inps > threshold)
    tru_pos = np.logical_and(thresh_predictions, targets).sum()
    pred_pos = thresh_predictions.sum()
    targ_pos = targets.sum()

    if pred_pos > 0 and targ_pos > 0 and tru_pos > 0:
        precision = tru_pos / pred_pos
        recall = tru_pos / targ_pos
        F1 = (1 + beta_squared) * (precision * recall) / ((precision * beta_squared) + recall)
    else:
        F1 = 0

    return F1

def multilabel_thresholding(output, target, num_trials=1000, beta=1):
    output = output.cpu().detach().numpy().reshape(-1)
    target = target.cpu().detach().numpy().reshape(-1)

    thresholds = np.linspace(min(output), max(output), num_trials)

    F1s = np.zeros(thresholds.shape)
    for i, t in enumerate(thresholds):
        F1s[i] = f_measure(output, target, threshold=t, beta=beta)

    best_thresh = thresholds[np.argmax(F1s)]
    best_score = np.max(F1s)

    return best_score, best_thresh


class TestResults(object):

    def __init__(self, threshes):
        self.threshes = threshes
        self.outputs = np.array([])
        self.targets = np.array([])
        self.results_dict = {'t_pos': [], 't_neg': [], 'f_pos': [], 'f_neg': []}

    def update(self, output, target):

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        self.outputs = np.concatenate([self.outputs, output.reshape(-1)])
        self.targets = np.concatenate([self.targets, target.reshape(-1).astype(int)])

    def calculate_stats(self):

        r = {x:{} for x in ['precision', 'recall', 'true negative rate', 'prop_positive_predictions', 'prop_positive_targets']}
        for t in self.threshes:
            thresh_res = self.calculate_stats_for_thresh(t)
            for k in thresh_res.keys():
                r[k][t] = (thresh_res[k])
        return r

    def calculate_stats_for_thresh(self, thresh):

        predictions = (self.outputs > thresh)

        cat_preds = predictions.reshape(-1).astype('bool')
        cat_target = self.targets.reshape(-1).astype('bool')

        t_pos = (cat_preds & cat_target).sum()

        t_neg = (~cat_preds & ~cat_target).sum()

        f_pos = (cat_preds & ~cat_target).sum()

        f_neg = (~cat_preds & cat_target).sum()

        prop_positive_predictions = np.sum(cat_preds) / len(cat_preds)
        prop_positive_targets = np.sum(cat_target) / len(cat_target)


        r = {}
        r['precision'] = t_pos / (t_pos + f_pos) if (t_pos + f_pos) > 0 else 0
        r['recall'] = t_pos / (t_pos + f_neg) if (t_pos + f_neg) > 0 else 0
        r['true negative rate'] = t_neg / (t_neg + f_pos) if (t_neg + f_pos) > 0 else 0
        r['prop_positive_predictions'] = prop_positive_predictions
        r['prop_positive_targets'] = prop_positive_targets
        return r

    def make_pr_curve(self):
        # necessary to downsample PR graphs because wandb only does graphs of 10000 pts
        precision, recall, thresholds = precision_recall_curve(self.targets, self.outputs)
        # downsample_amt = (len(precision) // 10001) + 1
        # precision = precision[::downsample_amt]
        # recall = recall[::downsample_amt]
        # thresholds = thresholds[::downsample_amt]

        return precision, recall, thresholds


if __name__ == '__main__':
    num_trials = 100

    tr = TestResults(threshes=[0.1, 0.5, 0.9])

    for x in range(10):
        targets = torch.tensor(np.random.randint(0, 2, num_trials))
        outputs = (targets * 2 - 1) + torch.tensor(np.random.normal(0, 3, num_trials))
        # targets = torch.round(outputs)
        tr.update(torch.sigmoid(outputs), targets)

    asdf = find_thresh_for_given_recall(torch.Tensor(tr.outputs), torch.Tensor(tr.targets), target_recall=0.33)

    res = tr.calculate_stats()
    p, r, t = tr.make_pr_curve()
