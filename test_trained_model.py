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

model_path = r'transformer_2020-09-21 18-12_epoch-4_24.24.1.pt'

dset_path = params.dset_path

dset_tr = dl.MonoFolkSongDataset(
    dset_fname=params.dset_path,
    seq_length=params.seq_length,
    base='train',
    num_dur_vals=params.num_dur_vals,
    proportion_for_stats=params.proportion_for_stats)
dloader = DataLoader(dset_tr, batch_size=params.batch_size, pin_memory=True)
num_feats = dset_tr.num_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tfsm.TransformerModel(
    num_feats,
    feedforward_size=params.ninp,
    hidden=params.nhid,
    nlayers=params.nlayers,
    dropout=params.dropout).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(model_path, map_location=device)
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

model.eval()

for i, batch in enumerate(dloader):
    batch = batch.float().to(device)
    input, target, batch = prepare_batch(batch)
    break

output = model(input, batch)

ind_rand = np.random.choice(output.shape[1])
fig, axs = po.plot(output, target, ind_rand, params.num_dur_vals, input)
fig.show()
