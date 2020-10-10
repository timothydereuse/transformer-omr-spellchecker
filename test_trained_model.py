import torch
import transformer_encoder_model as tem
import data_loaders as dl
import plot_outputs as po
import numpy as np
import make_supervised_examples as mse
from torch.utils.data import IterableDataset, DataLoader
import model_params as params
from importlib import reload

reload(params)
reload(mse)

def get_maxes(b, dset):
    pitch_out = b[:, :, :dset.pitch_subvector_len]
    dur_out = b[:, :, -dset.dur_subvector_len:]

    pitch_inds = pitch_out.argsort(dim=2, descending=True)
    dur_inds = dur_out.argsort(dim=2, descending=True)

    return pitch_inds, dur_inds


def accuracy_with_masks(out, tar, mask_inds=None):
    po, do = out
    pt, dt = tar
    if mask_inds is not None:
        y = mask_inds[:, 1]
        x = mask_inds[:, 0]
        po = po[x, y]
        do = do[x, y]
        pt = pt[x, y]
        dt = dt[x, y]

    pitch_correct = (po == pt).sum().detach().cpu().numpy()
    dur_correct = (do == dt).sum().detach().cpu().numpy()
    return pitch_correct, dur_correct


def eval_model(model, dset, device='cpu'):
    cmn = 5

    dloader = DataLoader(dset, batch_size=params.batch_size, pin_memory=True)

    model.eval()
    res = {}
    labels = ['global', 'mask', 'rand', 'left']
    for x in labels:
        res[x] = {'dur': [0] * cmn, 'pitch': [0] * cmn, 'count': 0}

    for batch in dloader:
        batch = batch.float().to(device)
        input, inds = mse.mask_indices(batch, **params.mask_indices_settings)
        mask_inds, rand_inds, left_inds = inds
        target = batch

        with torch.no_grad():
            output = model(input)

        tar_m = get_maxes(target, dset)
        tar_m = (tar_m[0][:, :, 0], tar_m[1][:, :, 0])
        out_m = get_maxes(output, dset)

        total_num_samples = output.shape[1] * output.shape[0]
        s = {'global': None, 'mask': mask_inds, 'rand': rand_inds, 'left': left_inds}

        for k in s:
            res[k]['count'] += total_num_samples if (s[k] is None) else len(s[k])
            for commonest in range(cmn):
                out_nth_m = (out_m[0][:, :, commonest], out_m[1][:, :, commonest])
                pf, df = accuracy_with_masks(out_nth_m, tar_m, s[k])
                res[k]['pitch'][commonest] += pf
                res[k]['dur'][commonest] += df

    for k in res.keys():
        res[k]['dur_acc'] = []
        res[k]['pitch_acc'] = []
        for c in range(cmn):
            res[k]['dur_acc'].append(sum(res[k]['dur'][:c+1]) / res[k]['count'])
            res[k]['pitch_acc'].append(sum(res[k]['pitch'][:c+1]) / res[k]['count'])

    return res, output


if __name__ == '__main__':
    model_path = r'trained_models\transformer_best_2020-10-07 15-02_ep-99_512.256.3.2.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dset_path = params.dset_path
    checkpoint = torch.load(model_path, map_location=device)

    dset_tr = dl.MonoFolkSongDataset(
        dset_fname=params.dset_path,
        seq_length=params.seq_length,
        base='test',
        num_dur_vals=params.num_dur_vals,
        use_stats=checkpoint['dset_stats'])
    dloader = DataLoader(dset_tr, batch_size=params.batch_size, pin_memory=True)

    model = tem.TransformerBidirectionalModel(
            num_feats=dset_tr.num_feats,
            d_model=params.d_model,
            hidden=params.hidden,
            nlayers=params.nlayers,
            nhead=params.nhead,
            dropout=params.dropout
            ).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # print(f'global: pitch_accuracy: {pitch_acc} | dur_acc: {dur_acc}')
    # print(f'mask: pitch_accuracy: {mask_pitch_acc} | dur_acc: {mask_dur_acc}')
    # print(f'rand: pitch_accuracy: {rand_pitch_acc} | dur_acc: {rand_dur_acc}')

    res_dict, output = eval_model(model, dset_tr, device)

    # ind_rand = np.random.choice(output.shape[1])
    # fig, axs = po.plot(output, target, ind_rand, params.num_dur_vals, input)
    # fig.show()
