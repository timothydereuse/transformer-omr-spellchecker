# transformer-midi-error-correction

Experiments in training a transformer network to identify "errors" (defined as any deviation from a ground truth) in symbolic music.

`data_loaders.py` contains a functionality for creating batches and extracting statistics from the data for training.

The main model is `transformer_full_g2p_model.py`. Variants of this model can be found in `/old_models`.

There is an older, simplified version of the transformer model with detailed annotations available as a notebook on [Google Colab](https://colab.research.google.com/drive/1Vzd3v-8HOTdQPmVC6gr0qzIwTSLJEb_6?usp=sharing).

## Preparing Data And Training

Data must be processed from `.krn` or `.musicxml` files into a `.hdf5` file. This processing is done by the `data_management/make_hdf5.py` script. It accepts a dictionary linking `corpus_names` to `directories.` The resulting `.hdf5` file will have each file indexed as a two-dimensional array of integers under `corpus_name/file_name.`

The model is trained by setting parameters in `model_params.py` and running `transformer_errors_train.py`.
