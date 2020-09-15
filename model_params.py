# -- paths to data
kern_paths = {
    'essen': r"D:\Documents\datasets\essen\europa",
    'meertens': r"D:\Documents\datasets\meertens_tune_collection\mtc-fs-1.0.tar\krn"
}
dset_path = r"essen_meertens_songs.hdf5"

# -- definition of transformer model structure
nhid = 64          # the dimension of the feedforward network
ninp = 64          # the dimension of the internal transformer representation
nlayers = 1         # number of encoder/decoder layers
num_dur_vals = 17   # number of duration values
seq_length = 50     # length of song sequences
proportion_for_stats = 1
dropout = 0.1       # dropout probability

# -- training parameters
num_epochs = 5
val_set_size = 0.1
lr = 0.0001
batch_size = 1000
lr_plateau_factor = 0.2
lr_plateau_patience = 3
lr_plateau_threshold = 0.001
clip_gradient_norm = 0.5
save_model_every = 10     # save a new model every X epochs
save_img_every = 2

# -- data augmentation
remove_indices_settings = {'mode': 'center', 'num_indices': 7}
