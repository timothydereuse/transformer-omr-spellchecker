import torch
import transformer_full_seq_model as tfsm
import data_loaders as dl
import plot_outputs as po
import numpy as np
from torch.utils.data import DataLoader

model_path = r'trained_models\transformer_epoch-45_256.256.2.pt'

dset_path = r"essen_meertens_songs.hdf5"
num_dur_vals = 15
seq_length = 60
num_errors = 3

num_epochs = 50
nhid = 256        # the dimension of the feedforward network
ninp = 256
nlayers = 2        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
dropout = 0.1      # the dropout value

dset_tr = dl.MonoFolkSongDataset(dset_path, seq_length, num_dur_vals=num_dur_vals, proportion_for_stats=1)
dloader = DataLoader(dset_tr, 50, pin_memory=True)
num_feats = dset_tr.num_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tfsm.TransformerModel(num_feats, ninp, nhid, nlayers, dropout).to(device)

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


model.eval()

for i, batch in enumerate(dloader):
    batch = torch.transpose(batch, 0, 1).float().to(device)
    edit_batch = batch.clone().detach()
    err_positions = tuple(np.random.choice(seq_length, num_errors, replace=False))
    edit_batch = add_errors_to_batch(edit_batch, err_positions)

    output = model(edit_batch, edit_batch)
    break

ind_rand = np.random.choice(output.shape[1])
fig, axs = po.plot(output, batch, ind_rand, num_dur_vals, errored=edit_batch)
fig.show()
