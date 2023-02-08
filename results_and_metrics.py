import torch
import torch.nn as nn
import plot_outputs as po
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader
import model_params as params
from importlib import reload
import data_management.vocabulary as vocab
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, average_precision_score

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

def matthews_correlation(outputs, targets, threshold=0.5):
    thresh_predictions = (outputs > threshold)

    mcc = matthews_corrcoef(thresh_predictions, targets)

    return mcc


def multilabel_thresholding(output, target, num_trials=2000):

    if type(output) is torch.Tensor:
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
    output = output.reshape(-1)
    target = target.reshape(-1)

    upper_positive = np.max(output[target == 1])
    lower_positive = np.min(output[target == 1])

    thresholds = np.linspace(lower_positive, upper_positive, num_trials)

    mccs = np.zeros(thresholds.shape)
    for i, t in enumerate(thresholds):
        mccs[i] = matthews_correlation(output, target, threshold=t)

    best_thresh = thresholds[np.argmax(mccs)]
    best_score = np.max(mccs)

    return best_score, best_thresh


def normalized_recall(outputs, targets):
    num_positive = np.sum(targets)
    n = outputs.shape[0]

    order = np.argsort(outputs * -1)
    ranks = np.argsort(order)
    positive_locs = np.nonzero(targets)

    sum_rank = np.sum(ranks[positive_locs])
    sum_i = (num_positive * (num_positive - 1)) // 2

    res = 1 - ((sum_rank - sum_i) / (num_positive * (n - num_positive)))
    return res

class TestResults(object):

    def __init__(self, threshes, target_recalls):
        self.threshes = threshes
        self.target_recalls = target_recalls + [0]
        self.outputs = np.array([])
        self.targets = np.array([])
        self.results_dict = {'t_pos': [], 't_neg': [], 'f_pos': [], 'f_neg': []}

    def update(self, output, target):

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        self.outputs = np.concatenate([self.outputs, output.reshape(-1)])
        self.targets = np.concatenate([self.targets, target.reshape(-1).astype(int)])

    def calculate_stats(self):
        categories = [
            'precision',
            'recall',
            'true negative rate',
            'prop_positive_predictions',
            'prop_positive_targets',
            'mcc'
            ]

        r = {x:{} for x in categories}
        for i, t in enumerate(self.threshes):
            thresh_res = self.calculate_stats_for_thresh(t)
            for k in thresh_res.keys():
                target_recall = self.target_recalls[i]
                r[k][target_recall] = (thresh_res[k])

        r['average_precision'] = self.average_precision()
        r['normalized_recall'] = normalized_recall(self.outputs, self.targets)
        
        return r

    def sigmoid_outputs(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        return sigmoid(self.outputs)

    def calculate_stats_for_thresh(self, thresh):
        
        predictions = (self.sigmoid_outputs() > thresh)

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
        r['mcc'] = matthews_correlation(cat_preds, cat_target, thresh)
        return r

    def make_pr_curve(self):
        # necessary to downsample PR graphs because wandb only does graphs of 10000 pts
        precision, recall, thresholds = precision_recall_curve(self.targets, self.outputs)
        # downsample_amt = (len(precision) // 10001) + 1
        # precision = precision[::downsample_amt]
        # recall = recall[::downsample_amt]
        # thresholds = thresholds[::downsample_amt]

        return precision, recall, thresholds

    def average_precision(self):
        return average_precision_score(self.targets, self.outputs)

    def normalized_recall(self):



        return res

if __name__ == '__main__':
    num_trials = 100
    from sklearn.metrics import PrecisionRecallDisplay
    import matplotlib.pyplot as plt

    tr = TestResults(threshes=[0.1, 0.5, 0.9], target_recalls=[0.1, 0.5, 0.9])

    for x in range(10):
        targets = torch.tensor(np.random.choice([0, 1], num_trials, p=[0.1, 0.9]))
        outputs = torch.linspace(0, 1, num_trials) + (0.15 * targets)
        # targets = torch.round(outputs)
        tr.update(outputs, targets)

    asdf = find_thresh_for_given_recall(torch.Tensor(tr.outputs), torch.Tensor(tr.targets), target_recall=0.33)

    res = tr.calculate_stats()
    p, r, t = tr.make_pr_curve()

    display = PrecisionRecallDisplay.from_predictions(tr.targets, tr.outputs, name="TestData")
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    # plt.show()