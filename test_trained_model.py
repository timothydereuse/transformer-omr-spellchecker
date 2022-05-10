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


reload(params)
reload(po)

def precision_recall(inps, targets, threshold=0.5):
    thresh_predictions = (inps > threshold)
    tru_pos = np.logical_and(thresh_predictions, targets).sum()
    pred_pos = thresh_predictions.sum()
    targ_pos = targets.sum()

    precision = tru_pos / pred_pos if pred_pos > 0 else 0
    recall = tru_pos / targ_pos if targ_pos > 0 else 0

    return precision, recall

def find_thresh_for_given_recall(output, target, num_trials=1000, target_recall=0.5):
    output = torch.sigmoid(output).cpu().detach().numpy().reshape(-1)
    target = target.cpu().detach().numpy().reshape(-1)

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
    output = torch.sigmoid(output).cpu().detach().numpy().reshape(-1)
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
        self.results_dict = {'t_pos': [], 't_neg': [], 'f_pos': [], 'f_neg': []}
        self.outputs = np.array([])
        self.targets = np.array([])

    def update(self, output, target):

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        self.outputs = np.concatenate([self.outputs, sigmoid(output.reshape(-1))])
        self.targets = np.concatenate([self.targets, target.reshape(-1).astype(int)])

    def calculate_stats(self):
        r = {x:[] for x in ['precision', 'recall', 'true positive rate', 'true negative rate']}
        for t in self.threshes:
            thresh_res = self.calculate_stats_for_thresh(t)
            for k in thresh_res.keys():
                r[k].append(thresh_res[k])
        return r

    def calculate_stats_for_thresh(self, thresh):

        predictions = (self.outputs > thresh)

        cat_preds = predictions.reshape(-1).astype('bool')
        cat_target = self.targets.reshape(-1).astype('bool')

        t_pos = (cat_preds & cat_target).sum()
        self.results_dict['t_pos'].append(t_pos)

        t_neg = (~cat_preds & ~cat_target).sum()
        self.results_dict['t_neg'].append(t_neg)

        f_pos = (cat_preds & ~cat_target).sum()
        self.results_dict['f_pos'].append(f_pos)

        f_neg = (~cat_preds & cat_target).sum()
        self.results_dict['f_neg'].append(f_neg)

        results_dict = {x: sum(self.results_dict[x]) for x in self.results_dict.keys()}
        r = {}
        r['precision'] = results_dict['t_pos'] / (results_dict['t_pos'] + results_dict['f_pos'])
        r['recall'] = results_dict['t_pos'] / (results_dict['t_pos'] + results_dict['f_neg'])
        r['true positive rate'] = results_dict['t_pos'] / (results_dict['t_pos'] + results_dict['f_neg'])
        r['true negative rate'] = results_dict['t_neg'] / (results_dict['t_neg'] + results_dict['f_pos'])
        return r


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


if __name__ == '__main__':
    model_path = r'trained_models\lstut_best_LSTUT_FELIX_TRIAL_0_(2022.03.14.20.21)_1-1-1-01-0-32-32.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint['params']

    v = vocab.Vocabulary(load_from_file=params.saved_vocabulary)

    lstut_settings = params.lstut_settings
    lstut_settings['vocab_size'] = v.num_words
    lstut_settings['seq_length'] = params.seq_length
    model = lstut.LSTUT(**lstut_settings).to(device)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.load_state_dict(checkpoint['model_state_dict'])
    # best_thresh = checkpoint['best_thresh']

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dloader):
            batch = batch.float().cpu()
            inp, target = prepare_batch(batch)

            batch = batch.to(device)
            inp = inp.to(device)
            target = target.to(device)

            output = tst_model(inp)
            results_dict = test_results(output, target, results_dict, best_thresh)

    for k in results_dict.keys():
        # for c in results_dict[k].keys():
        #     results_dict[k][c] = np.sum(results_dict[k][c])
        results_dict[k]['precision'] = results_dict[k]['t_pos'] / (results_dict[k]['t_pos'] + results_dict[k]['f_pos'])
        results_dict[k]['recall'] = results_dict[k]['t_pos'] / (results_dict[k]['t_pos'] + results_dict[k]['f_neg'])
        results_dict[k]['True positive rate'] = results_dict[k]['t_pos'] / (results_dict[k]['t_pos'] + results_dict[k]['f_neg'])
        results_dict[k]['True negative rate'] = results_dict[k]['t_neg'] / (results_dict[k]['t_neg'] + results_dict[k]['f_pos'])

    print(results_dict)
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    ind = 93
    fig, axs = po.plot_pianoroll_corrections(batch[ind], inp[ind], target[ind], output[ind], best_thresh)
    # fg.show()
    fig.savefig(f'./out_imgs/ind_{ind}.png', bbox_inches='tight')
