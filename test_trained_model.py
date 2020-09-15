import torch
import transformer_full_seq_model as tfsm
import data_loaders as dl
import plot_outputs as po
import numpy as np
import make_supervised_examples as mse
from torch.utils.data import DataLoader
import model_params as params

model_path = r'trained_models\transformer_epoch-20_256.256.3.pt'

dset_path = r"essen_meertens_songs.hdf5"

dset_tr = dl.MonoFolkSongDataset(
    dset_path,
    params.seq_length,
    num_dur_vals=params.num_dur_vals,
    proportion_for_stats=1)
dloader = DataLoader(dset_tr, batch_size=20, pin_memory=True)
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
    return input, target

model.eval()

for i, batch in enumerate(dloader):
    batch = batch.float().to(device)
    input, target = prepare_batch(batch)

    break

output = model(input, target)

ind_rand = np.random.choice(output.shape[1])
fig, axs = po.plot(output, target, ind_rand, params.num_dur_vals, input)
fig.show()
