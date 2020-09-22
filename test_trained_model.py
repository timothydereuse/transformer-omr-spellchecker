import torch
import transformer_full_seq_model as tfsm
import data_loaders as dl
import plot_outputs as po
import numpy as np
import make_supervised_examples as mse
from torch.utils.data import DataLoader
import model_params as params
from importlib import reload

reload(params)
reload(mse)

model_path = r'trained_models\transformer_2020-09-21 15-39_ep-50_512.128.2.pt'
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

model = tfsm.TransformerModel(
    num_feats,
    feedforward_size=params.ninp,
    hidden=params.nhid,
    nlayers=params.nlayers,
    dropout=params.dropout).to(device)

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


def prepare_batch(batch):
    input, target = mse.remove_indices(batch, **params.remove_indices_settings)
    input = input.transpose(1, 0)
    target = target.transpose(1, 0)
    batch = batch.transpose(1, 0)
    return input, target, batch


def get_maxes(b):
    pitch_out = b[:, :, :-params.num_dur_vals]
    dur_out = b[:, :, -params.num_dur_vals:]
    pitch_inds = pitch_out.max(2).indices
    dur_inds = dur_out.max(2).indices
    return pitch_inds, dur_inds


model.eval()
incorrect_dur = 0
incorrect_pitch = 0
total_dur = 0
total_pitch = 0
with torch.no_grad():
    for i, batch in enumerate(dloader):
        batch = batch.float().to(device)
        input, target, batch = prepare_batch(batch)

        output = model(input, input)

        inds = (input.sum(2) == 0)[:, 0].nonzero().view(-1)

        pitch_o, dur_o = get_maxes(output)
        pitch_t, dur_t = get_maxes(target)

        pitch_o = pitch_o[inds]
        dur_o = dur_o[inds]

        pitch_diff = (pitch_o != pitch_t).sum()
        dur_diff = (dur_o != dur_t).sum()
        incorrect_dur += dur_diff.numpy()
        incorrect_pitch += pitch_diff.numpy()

        total_dur += dur_o.numel()
        total_pitch += pitch_o.numel()

        print(incorrect_dur, incorrect_pitch)

dur_acc = 1 - incorrect_dur / total_dur
pitch_acc = 1 - incorrect_pitch / total_pitch

print(f'pitch_accuracy: {pitch_acc} | dur_acc: {dur_acc}')


ind_rand = np.random.choice(output.shape[1])
fig, axs = po.plot(output, batch, ind_rand, params.num_dur_vals, input)
fig.show()
