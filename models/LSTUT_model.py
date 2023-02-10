import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder, AttentionBuilder
from fast_transformers.attention.attention_layer import AttentionLayer
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from models.positional_encoding import PositionalEncoding
import time


class LSTUT(nn.Module):

    def __init__(self, vocab_size, seq_length, d_model, lstm_layers, tf_layers, tf_depth,
                 tf_heads, hidden_dim, ff_dim, output_feats=1, positional_encoding=False, dropout=0.15):
        super(LSTUT, self).__init__()

        self.seq_length = seq_length
        self.num_feats = 1
        self.output_feats = output_feats
        self.lstm_layers = lstm_layers
        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.tf_depth = tf_depth
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout = dropout

        if positional_encoding:
            self.positional_encoding = PositionalEncoding(self.hidden_dim, max_len=self.seq_length + 1)
        else:
            self.positional_encoding = False

        self.initial = nn.Embedding(self.vocab_size, self.d_model, padding_idx=1)

        if tf_layers > 0 and tf_depth > 0:
            self.encoder = self.make_tf_encoder()

        if lstm_layers > 0:
            self.lstm1 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)

        self.final_ff = nn.Linear(self.d_model, self.output_feats)
        self.layer_norm = nn.LayerNorm([self.seq_length, self.d_model], elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm([self.seq_length, self.d_model], elementwise_affine=False)

    def make_tf_encoder(self):

        layer_kwargs = {                   
                    'd_model': self.d_model,
                    'd_ff': self.ff_dim,
                    'dropout': self.dropout,
                    'activation': 'gelu'
                    }

        # this is necessary to get the package to work in 0.2.2 and 0.4.0, which is necessary
        # because compute canada doesn't play nice with 0.4.0 and my own laptop doesn't play
        # nice with 0.4.0 for some reason. i hate this :(
        try:
            att_builder = AttentionBuilder.from_kwargs(query_dimensions=self.hidden_dim)
        except ValueError:
            att_builder = AttentionBuilder()
            layer_kwargs['n_heads'] = self.tf_heads

        encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(att_builder.get('linear'),
                        d_model=self.d_model,
                        n_heads=self.tf_heads,
                        d_keys=self.hidden_dim,
                        d_values=self.hidden_dim,
                    ),
                    **layer_kwargs
                )
                for _ in range(self.tf_layers)
            ], None)
        return encoder

    def forward(self, src):

        x = self.initial(src)

        if self.lstm_layers > 0:
            self.lstm1.flatten_parameters()
            x, _ = self.lstm1(x)
            lstm1_out = x.clone()

        if self.positional_encoding:
            x = self.positional_encoding(x)

        if self.tf_layers > 0:
            for i in range(self.tf_depth):
                x = self.encoder(x)

        if self.lstm_layers > 0:
            x = self.layer_norm(x + lstm1_out)
            self.lstm2.flatten_parameters()
            x, _ = self.lstm2(x)

        x = self.layer_norm2(x)
        x = self.final_ff(x)

        return x


if __name__ == '__main__':

    batch_size = 10
    seq_length = 128
    num_feats = 1
    vocab_size = 100

    X = (torch.rand(batch_size, seq_length) * vocab_size).floor().type(torch.long)
    tgt = (torch.rand(batch_size, seq_length, 1) - 0.1).round()

    model = LSTUT(
        seq_length=seq_length,
        output_feats=num_feats,
        d_model=124,
        lstm_layers=0,
        tf_layers=1,
        tf_depth=2,
        tf_heads=2,
        hidden_dim=64,
        ff_dim=128,
        positional_encoding=False,
        vocab_size=vocab_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    res = model(X)

    # assert False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    sched = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, verbose=False)

    num_epochs = 200

    model.train()
    epoch_loss = 0
    for i in range(num_epochs):

        start_time = time.time()
        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        sched.step()

        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

    model.eval()

    x = AttentionLayer(
                        AttentionBuilder.from_kwargs(query_dimensions=64).get('linear'),
                        64,
                        4,
                        d_keys=32,
                        d_values=32,
                    )

    # def make_tf_encoder(self):

    #     encoder = TransformerEncoder(
    #         [
    #             TransformerEncoderLayer(
    #                 AttentionLayer(
    #                     AttentionBuilder.from_kwargs({} ).get('linear'),
    #                     d_model=54,
    #                     n_heads=4,
    #                     d_keys=32,
    #                     d_values=32,
    #                 ),
    #                 n_heads=4
    #                 d_model=54,
    #                 d_ff=128,
    #                 dropout=0.5,
    #                 activation='gelu',
    #             )
    #             for _ in range(2)
    #         ], None)
    #     return encoder