# name of this set of parameters:
params_name = 'TRIAL'

# -- definition of symbolic music representation
num_dur_vals = 0   # number of duration values
seq_length = 40     # length of song sequences
padding_amt = 5    # max padding on both sides of a song
proportion_for_stats = 1

# -- definition of transformer model structure
d_model = 256          # the dimension of the internal transformer representation
hidden = d_model * 2    # the dimension of the feedforward network
nlayers = 2             # number of encoder/decoder layers
nhead = 2               # number of attention heads
dropout = 0.1           # dropout probability

# -- data augmentation
mask_indices_settings = {
    'num_indices': int(seq_length * 0.15),
    'prob_random': 0.05,
    'prob_same': 0.15,
    'continguous': False
}

# -- training parameters
trial_run = True               # sets dataset to be comically small, for testing.
num_epochs = 20                # number of epochs to train for
lr = 0.0001                      # learning rate
batch_size = 2048               # size of each batch
lr_plateau_factor = 0.25
lr_plateau_patience = 5
lr_plateau_threshold = 0.002
clip_gradient_norm = 0.5
early_stopping_patience = 25    # abort training if it's been this long since best model
save_model_every = 1000         # save a new model every X epochs
save_img_every = 10              # save a new test image from the validation set every X epochs

# -- paths to data
raw_data_paths = {
    'essen': r"D:\Documents\datasets\essen\europa",
    'meertens': r"D:\Documents\datasets\meertens_tune_collection\mtc-fs-1.0.tar\krn"
}
dset_path = r"synthetic_repetition_dset.hdf5"
beat_multiplier = 48  # duration values will be multiplied by this number and then rounded
test_proportion = 0.1
validate_proportion = 0.1

# -- logging
import logging, datetime
start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
log_fname = f'./logs/training_{params_name}_{start_training_time}.log'
results_fname = f'./logs/test_results_{params_name}_{start_training_time}.log'

logging.basicConfig(filename=log_fname, filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
if not any([type(x) is logging.StreamHandler for x in logging.getLogger().handlers]):
    logging.getLogger().addHandler(logging.StreamHandler())

# -- constants that need to be here so that they can be referenced, but shouldn't be changed
flags = {
    'sos': [-1],
    'eos': [-2],
    'mask': [-3],
    'pad': [-4]}
if num_dur_vals > 0:
    for x in flags.keys():
        flags[x] = [0] + flags[x]

# deprecated
remove_indices_settings = {
    'mode': 'center',
    'num_indices': 1
}
