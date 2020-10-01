import torch
import transformer_encoder_model as tem
import data_loaders as dl
import plot_outputs as po
import numpy as np
import make_supervised_examples as mse
from torch.utils.data import DataLoader
import model_params as params
from importlib import reload

reload(params)
reload(mse)

model_path = r'trained_models\transformer_best_2020-09-30 15-19_ep-160_1024.256.6.4.pt'
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
num_feats = dset_tr.num_feats

model = tem.TransformerBidirectionalModel(
        num_feats=num_feats,
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


def add_errors_to_batch(inp, positions):
    inp[positions, :, :] = 1
    inp[positions, :, -1] = 1
    inp[positions, :, 1] = 1
    return inp


def get_maxes(b):
    pitch_out = b[:, :, :dset_tr.pitch_subvector_len]
    dur_out = b[:, :, -dset_tr.dur_subvector_len:]
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
    return pitch_diff.numpy(), dur_diff.numpy()

model.eval()
global_f_dur = 0
global_f_pitch = 0
global_total = 0
mask_f_dur = 0
mask_f_pitch = 0
mask_total = 0
rand_f_dur = 0
rand_f_pitch = 0
rand_total = 0
with torch.no_grad():
    for i, batch in enumerate(dloader):
        batch = batch.float().to(device)
        input, inds = mse.mask_indices(batch, **params.mask_indices_settings)
        mask_inds, rand_inds = inds
        target = batch

        output = model(input)

        out_m = get_maxes(output)
        tar_m = get_maxes(target)
        # inp_m = get_maxes(input)

        pf, df = accuracy_with_masks(out_m, tar_m)
        global_f_pitch += pf
        global_f_dur += df
        global_total += out_m[0].numel()

        pf, df = accuracy_with_masks(out_m, tar_m, mask_inds)
        mask_f_pitch += pf
        mask_f_dur += df
        mask_total += mask_inds.shape[0]

        pf, df = accuracy_with_masks(out_m, tar_m, rand_inds)
        rand_f_pitch += pf
        rand_f_dur += df
        rand_total += rand_inds.shape[0]

        print(pf, df)

dur_acc = 1 - global_f_dur / global_total
pitch_acc = 1 - global_f_pitch / global_total

mask_dur_acc = 1 - mask_f_dur / mask_total
mask_pitch_acc = 1 - mask_f_pitch / mask_total

rand_dur_acc = 1 - rand_f_dur / rand_total
rand_pitch_acc = 1 - rand_f_pitch / rand_total

print(f'global: pitch_accuracy: {pitch_acc} | dur_acc: {dur_acc}')
print(f'mask: pitch_accuracy: {mask_pitch_acc} | dur_acc: {mask_dur_acc}')
print(f'rand: pitch_accuracy: {rand_pitch_acc} | dur_acc: {rand_dur_acc}')

ind_rand = np.random.choice(output.shape[1])
fig, axs = po.plot(output, target, ind_rand, params.num_dur_vals, input)
fig.show()
