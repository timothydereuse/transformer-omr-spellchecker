import numpy as np
import torch
import agnostic_omr_dataloader as dl
from data_management.vocabulary import Vocabulary
from torch.utils.data import IterableDataset, DataLoader
import torch
from sklearn import preprocessing, decomposition, neighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import torch.nn as nn 
import training_helper_functions as tr_funcs
import time


EMBED_DIMENSION = 300 
EMBED_MAX_NORM = 1

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size):
        super(CBOW_Model, self).__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )

        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class CBOWTrainingDataGenerator(object):
    def __init__(self):
        pass

    def add_errors_to_batch(self, batch):
        seq_len = batch.shape[1]
        middle = (seq_len // 2) + 1

        targets = batch[:, middle]
        inputs = torch.concat((batch[:, :middle], batch[:, middle + 1:]), 1)

        return (inputs.long()), (targets.long())

if __name__ == '__main__':

    device, num_gpus = tr_funcs.get_cuda_info()
    v = Vocabulary(load_from_file='./data_management/vocab.txt')
    batch_size = 500000

    dset_kwargs = {
        'dset_fname': 'processed_datasets/all_string_quartets_agnostic_bymeasure.h5',
        'seq_length': 7,
        'padding_amt': 7,
        'dataset_proportion': 0.5,
        'vocabulary': v,
        'all_subsequences': True
    }

    dset_tr = dl.AgnosticOMRDataset(base='train', **dset_kwargs)
    dset_vl = dl.AgnosticOMRDataset(base='validate', **dset_kwargs)
    dset_tst = dl.AgnosticOMRDataset(base='test', **dset_kwargs)

    dloader = DataLoader(dset_tr, batch_size, pin_memory=True)
    dloader_val = DataLoader(dset_vl, batch_size, pin_memory=True)
    dloader_tst = DataLoader(dset_tst, batch_size, pin_memory=True)

    model = CBOW_Model(len(v.vtw))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    error_generator = CBOWTrainingDataGenerator()
    # inputs, targets = error_generator.add_errors_to_batch(batch)

    run_epoch_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'example_generator': error_generator,
    }

    val_losses = []
    train_losses = []

    for epoch in range(10):
        epoch_start_time = time.time()

        # perform training epoch
        model.train()
        train_loss, tr_exs = tr_funcs.run_epoch(
            dloader=dloader,
            train=True,
            log_each_batch=True,
            **run_epoch_kwargs
        )

        # test on validation set
        model.eval()
        num_entries = 0
        val_loss = 0.
        with torch.no_grad():
            val_loss, val_exs = tr_funcs.run_epoch(
                dloader=dloader_val,
                train=False,
                log_each_batch=False,
                **run_epoch_kwargs
            )

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        epoch_end_time = time.time()
        print(
            f'epoch {epoch:3d} | '
            f's/epoch    {(epoch_end_time - epoch_start_time):3.5e} | '
            f'train_loss {train_loss:1.6e} | '
            f'val_loss   {val_loss:1.6e} | '
        )
    
    embeddings = list(model.parameters())[0].detach().numpy()

    reduction = TSNE()

    reduced_embeddings = reduction.fit_transform(embeddings)


    import datetime

    start_training_time = datetime.datetime.now().strftime("(%Y.%m.%d.%H.%M)")
    model_path = f'./embedding_model_{start_training_time}.pt'
    torch.save(model, model_path)

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # vocab_words = [v.vtw[x] for x in sorted(list(v.vtw.keys()))]

    # fig.add_trace(
    #     go.Scatter(
    #         x=reduced_embeddings[:, 0],
    #         y=reduced_embeddings[:, 1],
    #         mode="text",
    #         text=vocab_words,
    #         textposition="middle center",
    #     )
    # )

    