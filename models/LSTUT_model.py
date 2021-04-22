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
                 n_heads, hidden_dim, ff_dim, tf_depth=3, dropout=0.15):
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

        encoder_builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            query_dimensions=self.hidden_dim,
            value_dimensions=self.hidden_dim,
            feed_forward_dimensions=self.ff_dim,
            attention_type='linear',
            dropout=dropout
        )

        self.initial_ff = nn.Linear(self.num_feats, self.d_model)
        self.lstm1 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)
        self.encoder = encoder_builder.get()
        self.lstm2 = nn.LSTM(self.d_model, self.d_model // 2, self.lstm_layers, batch_first=True, bidirectional=True)
        self.final_ff = nn.Linear(self.d_model, self.output_feats)

    def forward(self, src):

        x = self.initial_ff(src)

        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        for i in range(self.tf_depth):
            x = self.encoder(x)

        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(x)

        x = self.final_ff(x)

        return x


if __name__ == '__main__':

    import model_params

    params = model_params.Params(r'params_default.json', False, 0)

    batch_size = 10
    seq_len = 100
    output_pts = 1
    num_feats = 3

    X = torch.rand(batch_size, seq_len, num_feats)
    tgt = (torch.rand(batch_size, seq_len, output_pts) - 0.4).round()

    model = LSTUT(
        num_feats=num_feats,
        output_feats=output_pts,
        lstm_layers=3,
        n_layers=1,
        tf_depth=2,
        n_heads=2,
        hidden_dim=64,
        ff_dim=64)

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

# d_model = 32
# k = 5
# b = 3
# al = AttentionLayer(LinearAttention(d_model), d_model, 4, )
# S = torch.rand(k, d_model)
# S = S.unsqueeze(0).repeat(b, 1, 1)
# Z = torch.rand(b, 40, d_model)
#
# # gotta make a bunch of masks that mask nothing
# mask = FullMask(N=k, M=40)
# ql_mask = LengthMask(torch.ones(b) * k)
# kl_mask = LengthMask(torch.ones(b) * 40)
#
# res = al(S, Z, Z, mask, ql_mask, kl_mask)
