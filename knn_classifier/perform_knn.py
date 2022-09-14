import numpy as np
import torch
import agnostic_omr_dataloader as dl
from data_management.vocabulary import Vocabulary
from data_augmentation.error_gen_logistic_regression import ErrorGenerator
from torch.utils.data import IterableDataset, DataLoader
import torch
from sklearn import preprocessing, decomposition, neighbors
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn 
import training_helper_functions as tr_funcs
import time
from embedding_utils import CBOW_Model, CBOWTrainingDataGenerator, rolling_window
import matplotlib.pyplot as plt

window_size = 9
n_nearest_neighbors = 90
embedding_dim_reduce = 5
smoothing = 2000

model = torch.load("knn_classifier\embedding_model_(2022.08.31.22.43).pt")
error_generator = ErrorGenerator(
    models_fpath='./processed_datasets/quartet_omr_error_models_bymeasure.joblib',
    smoothing=smoothing,
    simple=True,
    simple_error_rate=(2/smoothing)
    )

v = Vocabulary(load_from_file='./data_management/vocab.txt')

dset_kwargs = {
    'dset_fname': 'processed_datasets/all_string_quartets_agnostic_byline.h5',
    'seq_length': 5,
    'padding_amt': 1,
    'dataset_proportion': 0.7,
    'vocabulary': v,
    'shuffle_files': True
}
dset_tr = dl.AgnosticOMRDataset(base='train', **dset_kwargs)

embedding_vectors = list(model.parameters())[0].detach().numpy()
SVD = TruncatedSVD(n_components=embedding_dim_reduce)
reduced_vectors = SVD.fit_transform(embedding_vectors)

context_separator = CBOWTrainingDataGenerator()

for i, x in enumerate(dset_tr.iter_file()):
    batch, names = x
    batch = batch.reshape(-1).numpy()
    
    errored_seq, error_indices = error_generator.add_errors_to_seq(batch)
    errored_seq_subsequences = rolling_window(errored_seq, window=window_size)
    inputs, targets = context_separator.add_errors_to_batch(torch.Tensor(errored_seq_subsequences))

    # reshape indices to 1d array and back to preserve integrity of input sample shape
    embedded_inputs = reduced_vectors[inputs.reshape(-1)].reshape(inputs.shape[0], inputs.shape[1], embedding_dim_reduce)
    embedded_inputs = embedded_inputs.reshape(embedded_inputs.shape[0], -1)
    embedded_targets = reduced_vectors[targets]

    knn = NearestNeighbors(n_neighbors=n_nearest_neighbors, metric='minkowski')
    knn.fit(embedded_inputs)

    distances, neighbs_indices = knn.kneighbors(embedded_inputs)
    closest = np.stack([targets[neighbs_indices[x]] for x in range(neighbs_indices.shape[0])], 0)

    diffs = []
    for c in closest:
        target_token = c[0]
        neighbor_tokens = c[1:]
        
        # how close is embedded_target to any of the embedded_neighbors?
        # find distribution of best fit for embedded_neighbors, then use
        # p-test of some sort to determine if embedded_target is within
        # 95 percentile range or something. i dont know man

        if target_token in neighbor_tokens:
            diff = 0
        else:
            embedded_neighbors = embedded_inputs[neighbor_tokens]
            embedded_target = embedded_inputs[target_token]
            embedded_mean_neighbor = np.mean(embedded_neighbors, 0)

            distances = np.linalg.norm(embedded_target - embedded_neighbors, axis=1)
            weights = 1 / np.arange(1, 1 + len(distances))


            # diff = np.median(distances ** 2)
            diff = np.average(distances, weights=weights)

        diffs.append(diff)
    
    plt.clf()
    plt.plot(diffs, '.', linewidth=1)
    plt.plot(error_indices, 'o', linewidth=1)
    plt.show()

    break