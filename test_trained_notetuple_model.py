import torch
import torch.nn as nn
import data_loaders as dl
import plot_outputs as po
import numpy as np
import make_supervised_examples as mse
from torch.utils.data import DataLoader
# import models.LSTUT_model as lstut
import model_params as params
from importlib import reload
import plot_outputs as po

reload(params)
reload(mse)
reload(po)


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

    thresholds = np.linspace(min(output), max(output), num_trials)

    F1s = np.zeros(thresholds.shape)
    for i, t in enumerate(thresholds):
        F1s[i] = f_measure(output, target, t)

    best_thresh = thresholds[np.argmax(F1s)]
    best_score = np.max(F1s)

    return best_score, best_thresh


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
    model_path = r'trained_models\lstut_best_2021-01-14 11-28_lstm-128-128-2_tf-256-256-128-4-5.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dset_path = params.dset_path
    checkpoint = torch.load(model_path, map_location=device)

    dset_tst = dl.MidiNoteTupleDataset(
        dset_fname=params.dset_path,
        seq_length=params.seq_length,
        base='test',
        padding_amt=params.padding_amt,
        trial_run=params.trial_run)
    dloader = DataLoader(dset_tst, params.batch_size, pin_memory=True)

    tst_model = lstut.LSTUTModel(**params.lstut_settings).to(device)
    tst_model = nn.DataParallel(tst_model, device_ids=list(range(torch.cuda.device_count())))
    tst_model.load_state_dict(checkpoint['model_state_dict'])
    best_thresh = checkpoint['best_thresh']

    results_dict = {
        x: {'t_pos': [], 't_neg': [], 'f_pos': [], 'f_neg': []}
        for x in ['replace', 'insert', 'delete']}

    def prepare_batch(batch):
        inp, target = mse.error_indices(batch, **params.error_indices_settings)
        return inp, target

    tst_model.eval()
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
