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
import results_and_metrics as ttm
from knn_classifier.embedding_utils import (
    CBOW_Model,
    CBOWTrainingDataGenerator,
    rolling_window,
)
from training_helper_functions import get_nice_results_string

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


# make test dataset
def make_big_dataset():
    large_sample_inputs = []
    large_sample_targets = []
    for i, x in enumerate(dset_tr.iter_file()):
        batch, names = x
        inds_to_choose = np.random.choice(
            batch.shape[0], 1 + batch.shape[0] // 100, False
        )
        filtered_batch = batch[inds_to_choose]
        res = [rolling_window(x, window=window_size) for x in filtered_batch]
        res = np.concatenate(res, 0)
        inputs, targets = context_separator.add_errors_to_batch(torch.Tensor(res))
        large_sample_inputs.append(inputs.numpy())
        large_sample_targets.append(targets.numpy())
    large_sample_inputs = np.concatenate(large_sample_inputs, 0)
    large_sample_targets = np.concatenate(large_sample_targets, 0)
    return large_sample_inputs, large_sample_targets


def test_knn_detection(
    batch,
    error_generator,
    embedding_vectors,
    window_size=3,
    n_nearest_neighbors=50,
    embedding_dim_reduce=5,
    knn_bypass_thresh=1,
    metric_order=2,
    smoothing=500,
    use_big_dset=False,
):

    SVD = TruncatedSVD(n_components=embedding_dim_reduce)
    reduced_vectors = SVD.fit_transform(embedding_vectors)
    # reduced_vectors = embedding_vectors

    context_separator = CBOWTrainingDataGenerator()
    error_generator.simple_error_rate = 2 / smoothing

    scores = []

    batch = batch.reshape(-1).numpy()

    # errored_seq, error_indices = error_generator.add_errors_to_seq(batch)
    # errored_seq_subsequences = rolling_window(errored_seq, window=window_size)

    roll_window_subsequences = rolling_window(batch, window=window_size)
    piece_inputs, piece_targets = context_separator.add_errors_to_batch(
        torch.Tensor(roll_window_subsequences)
    )

    embed_tokens = lambda x: reduced_vectors[x.reshape(-1)].reshape(
        x.shape[0], x.shape[1], embedding_dim_reduce
    )

    # reshape indices to 1d array and back to preserve integrity of input sample shape
    embd_piece_inputs = embed_tokens(piece_inputs).reshape(piece_inputs.shape[0], -1)

    # if use_big_dset:
    #     large_sample_inputs, large_sample_targets = make_big_dataset()

    #     embd_dset_inputs = embed_tokens(large_sample_inputs).reshape(
    #         large_sample_inputs.shape[0], -1
    #     )
    #     embd_dset_targets = reduced_vectors[large_sample_targets]

    #     embd_all_inputs = np.concatenate([embd_piece_inputs, embd_dset_inputs], 0)
    #     # embd_all_targets = np.concatenate([embd_piece_targets, embd_dset_targets], 0)
    #     all_targets = np.concatenate([piece_targets, large_sample_targets], 0)

    embd_all_inputs = embd_piece_inputs
    # embd_all_targets = embd_piece_targets
    all_targets = piece_targets

    knn = NearestNeighbors(
        n_neighbors=n_nearest_neighbors, metric="minkowski", p=metric_order
    )
    knn.fit(embd_all_inputs)
    distances, neighbs_indices = knn.kneighbors(embd_piece_inputs)

    # is each point the same as points that have similar context to it?

    closest = np.stack(
        [all_targets[neighbs_indices[x]] for x in range(neighbs_indices.shape[0])],
        0,
    )
    diffs = process_closest(closest, reduced_vectors, knn_bypass_thresh)
    # best_score, best_thresh = ttm.multilabel_thresholding(diffs, error_indices)
    # diffs.append(diffs)
    # scores.append(best_score)

    return (diffs, piece_targets)


if __name__ == "__main__":

    window_size = 3
    n_nearest_neighbors = 10
    embedding_dim_reduce = 5

    error_generator = ErrorGenerator(
        error_models_fpath="processed_datasets\paired_quartets_error_model_bymeasure.joblib",
        smoothing=2,
        simple=True,
        simple_error_rate=(0.1),
    )
    error_generator.simple_probs = [0, 0.5, 0.5]

    v = Vocabulary(load_from_file="./processed_datasets/vocab_big.txt")

    dset_kwargs = {
        "dset_fname": r"processed_datasets\supervised_omr_targets_bymeasure.h5",
        "seq_length": 10,  # N.B. this does not matter
        "padding_amt": 1,
        "minibatch_div": 1,
        "vocabulary": v,
        "shuffle_files": True,
    }

    dset_test = dl.AgnosticOMRDataset(base="train/onepass", **dset_kwargs)
    embedding_vectors = np.load(r"knn_classifier\agnostic_embedding_vectors.npy")
    test_res = ttm.TestResults(threshes=[0.5], target_recalls=[0.9, 0.99])

    for x in dset_test.iter_file():
        print(f"parsing {x[1][0][-1]}")
        batch = x[0][:, 0, :]
        targets = x[0][:, 1, :]

        diffs, vector_targets = test_knn_detection(
            batch,
            error_generator,
            embedding_vectors,
            embedding_dim_reduce=embedding_dim_reduce,
            window_size=window_size,
            n_nearest_neighbors=n_nearest_neighbors,
        )

        test_res.update(diffs, targets)

    test_res.update_threshes_for_given_recalls()

    res_stats = test_res.calculate_stats()

    mcc, f1_thresh = ttm.multilabel_thresholding(test_res.outputs, test_res.targets)

    res_stats["average_precision"] = test_res.average_precision()
    res_stats["normalized_recall"] = ttm.normalized_recall(
        test_res.outputs, test_res.targets
    )
    res_stats["threshes"] = test_res.threshes
    res_stats["max_mcc"] = mcc

    res_string = get_nice_results_string("knn", res_stats)

    precision, recalls, threshes = test_res.make_pr_curve()

    # import pandas as pd

    # df = pd.DataFrame(
    #     data={
    #         "precision": precision,
    #         "recall": recalls,
    #         "threshes": np.concatenate([threshes, [threshes[-1]]]),
    #     }
    # )

    # df.to_csv("./results_csv/knn_PR_curve.csv")

    # error_indices_viz = np.nonzero(error_indices)[0]
    # diffs = np.array(diffs)
    # plt.clf()
    # plt.scatter(range(len(diffs)), diffs, s=10, c='b')
    # plt.scatter(error_indices_viz, diffs[error_indices_viz], s=30, c='r')
    # plt.show()
