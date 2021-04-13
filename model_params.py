# name of this set of parameters:
params_name = 'TESTING_POINTSET'

# -- definition of symbolic music representation
seq_length = 64     # length of song sequences
padding_amt = 4    # max padding on both sides of a song

# -- training parameters
trial_run = False
dataset_proportion = 0.2       # reduces size of dataset
num_epochs = 50                # number of epochs to train for
lr = 0.001                      # initial learning rate
batch_size = 256            # size of each batch
clip_gradient_norm = 0.5        # clip norm of gradient after each backprop
early_stopping_patience = 20    # abort training if it's been this long since best model
save_model_every = 30         # save a new model every X epochs
save_img_every = 3              # save a new test image from the validation set every X epochs
num_feats = 3

# -- definition of set transformer model
set_transformer_settings = {
    'num_feats': num_feats,
    'num_output_points': 40,
    'n_layers': 1,
    'n_heads': 4,
    'tf_depth': (5, 2),
    'hidden_dim': 128,
    'ff_dim': 256,
    'dropout': 0.1
}

if trial_run:
    dataset_proportion = 0.001
    set_transformer_settings = {
        'num_feats': num_feats,
        'num_output_points': 20,
        'n_layers': 1,
        'n_heads': 2,
        'tf_depth': 2,
        'hidden_dim': 32,
        'ff_dim': 32,
        'dropout': 0.1
    }

# -- data augmentation
error_indices_settings = {
    'num_insertions': 12,
    'num_deletions': 0,
    'num_replacements': 2
}

# -- learning rate plateau scheduler settings
scheduler_settings = {
    'factor': 0.25,
    'patience': 3,
    'threshold': 0.001,
    'verbose': True
}

# -- logging
import logging, datetime
start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
log_fname = f'./logs/training_{params_name}_{start_training_time}.log'
results_fname = f'./logs/test_results_{params_name}_{start_training_time}.log'

logging.basicConfig(filename=log_fname, filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
if not any([type(x) is logging.StreamHandler for x in logging.getLogger().handlers]):
    logging.getLogger().addHandler(logging.StreamHandler())

lstut_summary_str = (
    'lstm-{lstm_inp}-{lstm_hidden}-{lstm_layers}_tf-'
    '{tf_inp}-{tf_ff}-{tf_k}-{nhead}-{tf_depth}').format(**lstut_settings)

# -- constants that need to be here so that they can be referenced, but shouldn't be changed
flags = {
    'sos': [-1],
    'eos': [-2],
    'mask': [-3],
    'pad': [-4]}

notetuple_flags = {
    'sos': [0, 0, 0, 10],
    'eos': [0, 0, 0, 20],
    'mask': [0, 0, 0, 30],
    'pad': [0, 0, 0, 0]
}

# deprecated
remove_indices_settings = {
    'mode': 'center',
    'num_indices': 1
}

raw_data_paths = {
    'essen': r"D:\Documents\datasets\essen\europa",
    'meertens': r"D:\Documents\datasets\meertens_tune_collection\mtc-fs-1.0.tar\krn"
}

# -- definition of autoregressive transformer model
transformer_ar_settings = {
    'input_feats': 1,
    'output_feats': 1,
    'n_layers': 2,
    'n_heads': 1,
    'hidden_dim': 32,
    'ff_dim': 32,
    'tf_depth': 1,
    'dropout': 0.15
}

# -- definition of LSTUT model
lstut_settings = {
    'seq_length': seq_length,
    'num_feats': 4,
    'lstm_inp': 128,
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'tf_inp': 256,
    'tf_ff': 256,
    'tf_k': 128,
    'nhead': 4,
    'tf_depth': 5,
    'dim_out': 3,
    'dropout': 0.15
}

# -- definition of LSTM model
lstm_settings = {
    'num_feats': 3,
    'lstm_inp': 128,
    'lstm_hidden': 128,
    'lstm_layers': 4,
    'dropout': 0.15
}

mask_indices_settings = {
    'num_indices': int(seq_length * 0.05),
    'prob_random': 0.00,
    'prob_same': 0.15,
    'continguous': False
}
