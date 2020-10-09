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
    pitch_inds = pitch_out.max(2).indices
    dur_inds = dur_out.max(2).indices
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

    pitch_diff = (po != pt).sum()
    dur_diff = (do != dt).sum()
    return pitch_diff.detach().cpu().numpy(), dur_diff.detach().cpu().numpy()


def eval_model(model, dset, device='cpu'):

    dloader = DataLoader(dset, batch_size=params.batch_size, pin_memory=True)

    model.eval()
    res = {}
    labels = ['global', 'mask', 'rand', 'left']
    for x in labels:
        res[x] = {'dur': 0, 'pitch': 0, 'count': 0}

    with torch.no_grad():
        for batch in dloader:
            batch = batch.float().to(device)
            input, inds = mse.mask_indices(batch, **params.mask_indices_settings)
            mask_inds, rand_inds, left_inds = inds
            target = batch

            output = model(input)

            out_m = get_maxes(output, dset)
            tar_m = get_maxes(target, dset)

            s = {'global': None, 'mask': mask_inds, 'rand': rand_inds, 'left': left_inds}
            for k in s:
                pf, df = accuracy_with_masks(out_m, tar_m, s[k])
                res[k]['pitch'] += pf
                res[k]['dur'] += df
                res[k]['count'] += out_m[0].numel() if (s[k] is None) else len(s[k])

    for k in res.keys():
        res[k]['dur_acc'] = 1 - (res[k]['dur'] / res[k]['count'])
        res[k]['pitch_acc'] = 1 - (res[k]['pitch'] / res[k]['count'])


    # res_dict = {
    #     'dur_acc': 1 - global_f_dur / global_total,
    #     'pitch_acc': 1 - global_f_pitch / global_total,
    #     'mask_dur_acc': 1 - mask_f_dur / mask_total,
    #     'mask_pitch_acc': 1 - mask_f_pitch / mask_total,
    #     'rand_dur_acc': 1 - rand_f_dur / rand_total,
    #     'rand_pitch_acc': 1 - rand_f_pitch / rand_total
    # }

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
