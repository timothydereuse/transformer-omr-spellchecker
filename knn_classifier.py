import numpy as np
import torch
import agnostic_omr_dataloader as dl
from data_management.vocabulary import Vocabulary
from torch.utils.data import IterableDataset, DataLoader
from sklearn import preprocessing, decomposition, neighbors
from metric_learn import NCA, LMNN, LFDA


def knn_training_data_from_batch(batch):
    seq_len = batch.shape[1]
    middle = (seq_len // 2) + 1

    targets = batch[:, middle]
    inputs = np.concatenate((batch[:, :middle], batch[:, middle + 1:]), 1)

    return inputs, targets

if __name__ == '__main__':

    test_amt = 5000
    k = 13
    fname = 'processed_datasets/all_string_quartets_agnostic.h5'
    seq_len = 5
    proportion = 0.02
    v = Vocabulary(load_from_file='./data_management/vocab.txt')
    dset = dl.AgnosticOMRDataset(fname, seq_len, v, dataset_proportion=proportion, shuffle_files=False, all_subsequences=True)

    dload = DataLoader(dset, batch_size=250)
    batches = []
    for i, x in enumerate(dset.iter_file()):
        print(i, x[0].shape)
        batch, names = x
        batches.append(batch)
        if i > 15:
            break

    batches = torch.cat(batches)

    # enc = preprocessing.OneHotEncoder(sparse=True, handle_unknown="ignore", min_frequency=100)
    # enc.fit(inputs)

    # # inputs are the neighborhoods of notes - targets are the notes themselves
    # onehot_inputs = enc.transform(inputs)
    # print(onehot_inputs.shape)

    # onehot_inp_train = onehot_inputs[:test_amt]
    # onehot_inp_test = onehot_inputs[test_amt:]

    inp_train = inputs[:test_amt]
    inp_test = inputs[test_amt:]
    targ_train = targets[:test_amt]
    targ_test = targets[test_amt:]

    # dim_reduce = decomposition.TruncatedSVD(n_components=20)
    # dim_reduce.fit(onehot_inp_train)
    # onehot_inp_train_reduced = dim_reduce.transform(onehot_inp_train)
    # onehot_inp_test_reduced = dim_reduce.transform(onehot_inp_test)

    # print('transforming...')
    # onehot_inp_train_reduced = nca.transform(onehot_inp_train)
    # onehot_inp_test_reduced = nca.transform(onehot_inp_test)

    knn = neighbors.NearestNeighbors(n_neighbors=k, metric="hamming")
    knn.fit(inp_train, targ_train)

    predictions = knn.kneighbors(inp_test, return_distance=False)
    prediction_inds = targ_test[predictions]
    
    targ_test_repmat = np.repeat(targ_test.reshape(-1, 1), k, 1)
    res = np.any(targ_test_repmat == prediction_inds, 1)

    print(np.mean(res))