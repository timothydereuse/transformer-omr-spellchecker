import sys
from os import path

# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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
import datetime
from knn_classifier.embedding_utils import CBOW_Model, CBOWTrainingDataGenerator

if __name__ == "__main__":

    device, num_gpus = tr_funcs.get_cuda_info()
    v = Vocabulary(load_from_file=r"processed_datasets\vocab_big.txt")
    batch_size = 50000
    base_lr = 0.005
    max_lr = 0.1
    ngram_length = 3
    early_stopping_patience = 5
    max_num_epochs = 1000
    minibatch_div = 20
    model_output_name = r"knn_classifier/embedding_n3_byline"
    dset_fname = r"processed_datasets\all_quartets_bymeasure.h5"

    dset_kwargs = {
        "dset_fname": dset_fname,
        "seq_length": ngram_length * 2 + 1,
        "padding_amt": 7,
        "minibatch_div": minibatch_div,
        "vocabulary": v,
        "all_subsequences": True,
    }

    dset_tr = dl.AgnosticOMRDataset(base="train", **dset_kwargs)
    dset_vl = dl.AgnosticOMRDataset(base="validate", **dset_kwargs)
    dset_tst = dl.AgnosticOMRDataset(base="test", **dset_kwargs)

    dloader = DataLoader(dset_tr, batch_size, pin_memory=True)
    dloader_val = DataLoader(dset_vl, batch_size, pin_memory=True)
    dloader_tst = DataLoader(dset_tst, batch_size, pin_memory=True)

    model = CBOW_Model(len(v.vtw))
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=base_lr, max_lr=max_lr, verbose=True, cycle_momentum=False
    )
    criterion = nn.CrossEntropyLoss()
    error_generator = CBOWTrainingDataGenerator()
    # inputs, targets = error_generator.add_errors_to_batch(batch)

    run_epoch_kwargs = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device,
        "example_generator": error_generator,
        "scheduler": scheduler,
    }

    val_losses = []
    train_losses = []

    for epoch in range(max_num_epochs):
        epoch_start_time = time.time()

        # perform training epoch
        model.train()
        train_loss, tr_exs = tr_funcs.run_epoch(
            dloader=dloader, train=True, log_each_batch=True, **run_epoch_kwargs
        )

        # test on validation set
        model.eval()
        num_entries = 0
        val_loss = 0.0
        with torch.no_grad():
            val_loss, val_exs = tr_funcs.run_epoch(
                dloader=dloader_val,
                train=False,
                log_each_batch=False,
                **run_epoch_kwargs,
            )

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        epoch_end_time = time.time()
        print(
            f"epoch {epoch:3d} | "
            f"s/epoch    {(epoch_end_time - epoch_start_time):3.5e} | "
            f"train_loss {train_loss:1.6e} | "
            f"val_loss   {val_loss:1.6e} | "
        )

        # early stopping
        time_since_best = epoch - val_losses.index(min(val_losses))
        if time_since_best > early_stopping_patience:
            print(
                f"stopping early at epoch {epoch} because validation score stopped increasing"
            )
            break

    print(train_losses, "\n\n", val_losses)

    embeddings = list(model.parameters())[0].detach().numpy()

    # reduction = TSNE()
    # reduced_embeddings = reduction.fit_transform(embeddings)
    start_training_time = datetime.datetime.now().strftime("(%Y.%m.%d.%H.%M)")
    model_path = f"./{model_output_name}_{start_training_time}.pt"
    torch.save(model, model_path)

    asdf = model.embeddings.weight.detach().numpy()
    np.save("knn_classifier/agnostic_embedding_vectors.npy", asdf)

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
