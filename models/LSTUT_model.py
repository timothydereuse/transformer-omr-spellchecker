import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.attention.attention_layer import AttentionLayer
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
# from positional_encoding import PositionalEncoding
import time


class LSTUT(nn.Module):

    def __init__(self, num_feats, output_feats, lstm_layers, n_layers,
                 n_heads, hidden_dim, ff_dim, tf_depth=3, vocab_size=0, dropout=0.15):
        super(LSTUT, self).__init__()

        self.num_feats = num_feats
        self.output_feats = output_feats
        self.lstm_layers = lstm_layers
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.tf_depth = tf_depth
        self.d_model = self.hidden_dim * self.n_heads
        self.vocab_size = vocab_size

        if vocab_size and num_feats > 1:
            raise ValueError("can't have multiple features and an embedding")

        encoder_builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
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

        self.lstm1 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)
        self.encoder = encoder_builder.get()
        self.lstm2 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)
        self.final_ff = nn.Linear(self.d_model, self.output_feats)

    def forward(self, src):

        x = self.initial(src)

        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        for i in range(self.tf_depth):
            x = self.encoder(x)

        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(x)

        x = self.final_ff(x)

        return x


if __name__ == '__main__':

    batch_size = 10
    seq_len = 128
    output_pts = 1
    num_feats = 1
    vocab_size = 100

    X = (torch.rand(batch_size, seq_len) * vocab_size).floor().type(torch.long)
    tgt = (torch.rand(batch_size, seq_len, output_pts) - 0.3).round()

    model = LSTUT(
        num_feats=num_feats,
        output_feats=output_pts,
        lstm_layers=3,
        n_layers=1,
        tf_depth=2,
        n_heads=2,
        hidden_dim=32,
        ff_dim=32,
        vocab_size=vocab_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    res = model(X)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    sched = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, verbose=False)

    num_epochs = 100

    model.train()
    epoch_loss = 0
    for i in range(num_epochs):

        start_time = time.time()
        optimizer.zero_grad()

        output = model(X)

        loss = criterion(tgt, output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        sched.step()

        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

    model.eval()