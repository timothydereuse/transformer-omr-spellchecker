import torch
import torch.nn as nn
import numpy as np

def rolling_window(a, window, pad=2):

    pad_amt = window // 2
    pad = np.repeat(pad, pad_amt)
    a = np.concatenate([pad, a, pad])

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