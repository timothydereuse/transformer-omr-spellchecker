import numpy as np
import torch
import agnostic_omr_dataloader as dl
from data_management.vocabulary import Vocabulary
from data_augmentation.error_gen_logistic_regression import ErrorGenerator
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import training_helper_functions as tr_funcs
import time
import test_trained_model as ttm
from knn_classifier.embedding_utils import CBOW_Model, CBOWTrainingDataGenerator, rolling_window
# import matplotlib.pyplot as plt
import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def process_closest(closest, reduced_vectors, knn_bypass_thresh):
    diffs = []
    for c in closest:
        target_token = c[0]
        neighbor_tokens = c[1:]
        
        # how close is embedded_target to any of the embedded_neighbors?
        # find distribution of best fit for embedded_neighbors, then use
        # p-test of some sort to determine if embedded_target is within
        # 95 percentile range or something. i dont know man

        match_pos = np.where(neighbor_tokens == target_token)[0]
        if len(match_pos) > 0 and match_pos[0] < knn_bypass_thresh:
            diffs.append(0)
            continue

        embedded_neighbors = reduced_vectors[neighbor_tokens]
        embedded_target = reduced_vectors[target_token]
        distances = np.linalg.norm(embedded_target - embedded_neighbors, axis=1, ord=2)
        # diff = np.min(distances)

        weights = (1 / np.arange(1, 1 + len(distances))) ** 2
        diff = np.average(distances, weights=weights)

        diffs.append(diff)
    return np.array(diffs)


def test_knn_detection(dset_tr, error_generator, embedding_vectors, window_size=3, n_nearest_neighbors=200,
    embedding_dim_reduce=9, knn_bypass_thresh=3, metric_order=2, smoothing=500, use_big_dset=False, pieces_to_try=25):

    SVD = TruncatedSVD(n_components=embedding_dim_reduce)
    reduced_vectors = SVD.fit_transform(embedding_vectors)

    context_separator = CBOWTrainingDataGenerator()
    error_generator.simple_error_rate = 2 / smoothing

    # make test dataset
    def make_big_dataset():
        large_sample_inputs = []
        large_sample_targets = []
        for i, x in enumerate(dset_tr.iter_file()):
            batch, names = x
            inds_to_choose = np.random.choice(batch.shape[0], 1 + batch.shape[0] // 100, False)
            filtered_batch = batch[inds_to_choose]
            res = [rolling_window(x, window=window_size) for x in filtered_batch]
            res = np.concatenate(res, 0)
            inputs, targets = context_separator.add_errors_to_batch(torch.Tensor(res))
            large_sample_inputs.append(inputs.numpy())
            large_sample_targets.append(targets.numpy())
        large_sample_inputs = np.concatenate(large_sample_inputs, 0)
        large_sample_targets = np.concatenate(large_sample_targets, 0)
        return large_sample_inputs, large_sample_targets

    scores = []
    for i, x in enumerate(dset_tr.iter_file()):
        batch, names = x
        batch = batch.reshape(-1).numpy()
        
        errored_seq, error_indices = error_generator.add_errors_to_seq(batch)
        errored_seq_subsequences = rolling_window(errored_seq, window=window_size)
        piece_inputs, piece_targets = context_separator.add_errors_to_batch(torch.Tensor(errored_seq_subsequences))

        embed_tokens = lambda x: reduced_vectors[x.reshape(-1)].reshape(x.shape[0], x.shape[1], embedding_dim_reduce)

        # reshape indices to 1d array and back to preserve integrity of input sample shape
        embd_piece_inputs = embed_tokens(piece_inputs).reshape(piece_inputs.shape[0], -1)
        embd_piece_targets = reduced_vectors[piece_targets]

        if use_big_dset:
            large_sample_inputs, large_sample_targets = make_big_dataset()

            embd_dset_inputs = embed_tokens(large_sample_inputs).reshape(large_sample_inputs.shape[0], -1)
            embd_dset_targets = reduced_vectors[large_sample_targets]

            embd_all_inputs = np.concatenate([embd_piece_inputs, embd_dset_inputs], 0)
            # embd_all_targets = np.concatenate([embd_piece_targets, embd_dset_targets], 0)
            all_targets = np.concatenate([piece_targets, large_sample_targets], 0)

        else: 
            embd_all_inputs = embd_piece_inputs
            # embd_all_targets = embd_piece_targets
            all_targets = piece_targets

        knn = NearestNeighbors(n_neighbors=n_nearest_neighbors, metric='minkowski', p=metric_order)
        knn.fit(embd_all_inputs)

        distances, neighbs_indices = knn.kneighbors(embd_piece_inputs)
        closest = np.stack([all_targets[neighbs_indices[x]] for x in range(neighbs_indices.shape[0])], 0)
        diffs = process_closest(closest, reduced_vectors, knn_bypass_thresh)
        best_score, best_thresh = ttm.multilabel_thresholding(diffs, error_indices, beta=0.5)
        scores.append(best_score)        
        if i >= pieces_to_try:
            break

    return scores

if __name__ == "__main__":
        
    window_size = 3
    n_nearest_neighbors = 200
    embedding_dim_reduce = 9
    knn_bypass_thresh = 3
    smoothing = 500
    use_big_dset = False
    pieces_to_try = 25

    error_generator = ErrorGenerator(
        models_fpath='./processed_datasets/quartet_omr_error_models_byline.joblib',
        smoothing=smoothing,
        simple=True,
        simple_error_rate=(2/smoothing)
        )
    error_generator.simple_probs = [0, 0.5, 0.5]

    v = Vocabulary(load_from_file='./data_management/vocab.txt')

    dset_kwargs = {
        'dset_fname': 'processed_datasets/all_string_quartets_agnostic_byline.h5',
        'seq_length': 100, # N.B. this does not matter because of reshape(-1) below
        'padding_amt': 1,
        'dataset_proportion': 1,
        'vocabulary': v,
        'shuffle_files': True
    }

    dset_tr = dl.AgnosticOMRDataset(base='train', **dset_kwargs)
    # model = torch.load(r"knn_classifier\agnostic_embedding_(2022.09.14.16.57).pt")
    # embedding_vectors = list(model.parameters())[0].detach().numpy()
    embedding_vectors = np.load(r'knn_classifier\agnostic_embedding_vectors_byline.npy')

    scores = test_knn_detection(dset_tr, error_generator, embedding_vectors, embedding_dim_reduce=15, pieces_to_try=2)
            # error_indices_viz = np.nonzero(error_indices)[0]
            # diffs = np.array(diffs)
            # plt.clf()
            # plt.scatter(range(len(diffs)), diffs, s=10, c='b')
            # plt.scatter(error_indices_viz, diffs[error_indices_viz], s=30, c='r')
            # plt.show()