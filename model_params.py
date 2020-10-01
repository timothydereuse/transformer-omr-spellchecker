# -- paths to data
raw_data_paths = {
    'essen': r"D:\Documents\datasets\essen\europa",
    'meertens': r"D:\Documents\datasets\meertens_tune_collection\mtc-fs-1.0.tar\krn"
}
dset_path = r"essen_meertens_songs.hdf5"
beat_multiplier = 48  # duration values will be multiplied by this number and then rounded
test_proportion = 0.1
validate_proportion = 0.1

# -- definition of symbolic music representation
num_dur_vals = 15   # number of duration values
seq_length = 40     # length of song sequences
padding_amt = 10    # max padding on both sides of a song
proportion_for_stats = 1

# -- definition of transformer model structure
d_model = 256           # the dimension of the internal transformer representation
hidden = d_model * 4    # the dimension of the feedforward network
nlayers = 6             # number of encoder/decoder layers
nhead = 4               # number of attention heads
dropout = 0.1           # dropout probability

# -- training parameters
trial_run = True                #
num_epochs = 161                 # number of epochs to train for
lr = 0.001                      # learning rate
batch_size = 1400               # size of each batch
lr_plateau_factor = 0.25
lr_plateau_patience = 4
lr_plateau_threshold = 0.01
clip_gradient_norm = 0.5
early_stopping_patience = 15    # abort training if no improvement after this many epochs
save_model_every = 40           # save a new model every X epochs
save_img_every = 5              # save a new test image from the validation set every X epochs


# -- data augmentation
remove_indices_settings = {
    'mode': 'center',
    'num_indices': 1
}
mask_indices_settings = {
    'num_indices': int(seq_length * 0.2),
    'prob_random': 0.15,
    'prob_same': 0.01,
    'continguous': False
}

# -- logging
import logging, datetime
start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
logging.basicConfig(filename='transformer_train.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
if not any([type(x) is logging.StreamHandler for x in logging.getLogger().handlers]):
    logging.getLogger().addHandler(logging.StreamHandler())

# -- constants that need to be here so that they can be referenced, but shouldn't be changed
flags = {
    'sos': [0, -1],
    'eos': [0, -2],
    'mask': [0, -3],
    'pad': [0, -4]
}
