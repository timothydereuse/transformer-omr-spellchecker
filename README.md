# transformer-omr-spellchecker

Experiments in training a transformer network to identify "errors" in symbolic music using a small corpus of training data. See the paper "A Transformer-based 'Spellchecker' for Detecting Errors in OMR Output," Timothy de Reuse and Ichiro Fujinaga, in ISMIR 2022 for a detailed description of our data augmentation method.

## Setup

Requires Python >= 3.8, and

```
numpy>=1.22.0
h5py>=3.0.0
sklearn>=1.0.0
torch>=1.10
numba>=0.56.0
pytorch-fast-transformers==0.2.2
music21>=7.0.0
```

## Preparing Data And Training

Data must be processed from `.krn` or `.musicxml` files (or any other symbolic music format parseable by `music21` that contains information from which a sensible engraving can be inferred) into several `.hdf5` file. We require a corpus of OMR'ed music files accompanied by their corrected versions (to serve as test and validation data) as well as a set of correct music files of similar genre to be used for training, through data augmentation (again, see the ISMIR paper cited above). The `run_all_data_preparation.py` script processes these files and produces five intermediary files:

- A `.h5` file containing only correct musical pieces in agnostic format (for training, w/data augmentation)
- A `.h5` file containing OMR'ed musical pieces aligned with corresponding correct musical files
- A `.h5` file containing supervised input / target training data based on alignments of the OMR'ed / correct musical pieces in the previous `.h5` file
- A `vocab.txt` file that assignes a single natural number index to each agnostic encoding token
- A `.joblib` file created by training a simple regression model on part of the OMR'ed musical pieces, for inserting OMR-like errors into the non-OMR'ed corpus of musical files

Versions of these are already included in the GitHub repository for this project under `/processed_datasets` and `/data_management`. If you want to make your own, you will have to change the names of the files in the `run_all_data_preparation.py` script to match the names of the folders on your local machine where your symbolic music files are kept.

The model is trained and tested by setting a parameters file and running `train_lstut_model.py` with the relevant arguments (run `python train_lstut_model.py -h` for details).
