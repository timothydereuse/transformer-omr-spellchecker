import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.attention.attention_layer import AttentionLayer
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
from models.positional_encoding import PositionalEncoding
import time


class LSTUT(nn.Module):

    def __init__(self, seq_length, num_feats, output_feats, lstm_layers, tf_layers,
                 tf_heads, hidden_dim, ff_dim, tf_depth=3, positional_encoding=False, vocab_size=0, dropout=0.15):
        super(LSTUT, self).__init__()

        self.seq_length = seq_length
        self.num_feats = num_feats
        self.output_feats = output_feats
        self.lstm_layers = lstm_layers
        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.tf_depth = tf_depth
        self.d_model = self.hidden_dim * self.tf_heads
        self.vocab_size = vocab_size

        if positional_encoding:
            self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.seq_length + 1)
        else:
            self.positional_encoding = False

        if vocab_size and num_feats > 1:
            raise ValueError("can't have multiple features and an embedding")

        encoder_builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.tf_layers,
            n_heads=self.tf_heads,
            query_dimensions=self.hidden_dim,
            value_dimensions=self.hidden_dim,
            feed_forward_dimensions=self.ff_dim,
            attention_type='linear',
            dropout=dropout
        )

        if self.num_feats == 1 and vocab_size > 0:
            self.initial = nn.Embedding(self.vocab_size, self.d_model, padding_idx=1)
        else:
            self.initial = nn.Linear(self.num_feats, self.d_model)

        if lstm_layers > 0:
            self.lstm1 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)

        if tf_layers > 0 and tf_depth > 0:
            self.encoder = encoder_builder.get()
        self.final_ff = nn.Linear(self.d_model, self.output_feats)
        self.layer_norm = nn.LayerNorm([self.seq_length, self.d_model])
        self.layer_norm2 = nn.LayerNorm([self.seq_length, self.d_model])

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
    output_pts = 1
    num_feats = 1
    vocab_size = 100

    X = (torch.rand(batch_size, seq_length) * vocab_size).floor().type(torch.long)
    tgt = (torch.rand(batch_size, seq_length, output_pts) - 0.1).round()

    model = LSTUT(
        seq_length=seq_length,
        num_feats=num_feats,
        output_feats=output_pts,
        lstm_layers=0,
        tf_layers=1,
        tf_depth=2,
        tf_heads=2,
        hidden_dim=64,
        ff_dim=64,
        positional_encoding=True,
        vocab_size=vocab_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    res = model(X)

    # assert False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    sched = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.25, verbose=False)

    num_epochs = 20

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