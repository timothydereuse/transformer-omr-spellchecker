# transformer-midi-error-correction

Experiments in training a transformer network to identify "errors" (defined as any deviation from a ground truth) in symbolic music.

`make_hdf5.py` processes datasets of .krn files into a MIDI-like event-based format. The `essen_meertens_songs.hdf5` is a selection of monophonic folk songs that have been processed with this script.

`factorizations.py` contains methods to turn the MIDI-like event-based format into various other forms, ready to be used as training data (in particular, a runlength encoding; for details, check "Coupled Recurrent Models for Polyphonic Music Composition," Thickstun et al., Proc. ISMIR 2019.)

`data_loaders.py` contains a functionality for creating batches and extracting statistics from the data for training.

The main model is `transformer_full_g2p_model.py`. Variants of this model can be found in `/old_models`.

There is an older, simplified version of the transformer model with detailed annotations available as a notebook on [Google Colab](https://colab.research.google.com/drive/1Vzd3v-8HOTdQPmVC6gr0qzIwTSLJEb_6?usp=sharing).

## Preparing Data And Training

Data must be processed from `.krn` or `.midi` files into a `.hdf5` file. This processing is done by the `data_management/make_hdf5.py` script. It accepts a dictionary linking `corpus_names` to `directories.` The resulting `.hdf5` file will have each file indexed as a two-dimensional array of integers under `corpus_name/file_name.`

The model is trained by setting parameters in `model_params.py` and running `transformer_errors_train.py`.
