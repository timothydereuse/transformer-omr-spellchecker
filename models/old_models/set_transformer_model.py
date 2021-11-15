import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.attention.attention_layer import AttentionLayer
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
# from positional_encoding import PositionalEncoding
import time


class SetTransformer(nn.Module):

    def __init__(self, num_feats, num_output_points, lstm_layers, n_layers,
                 n_heads, hidden_dim, ff_dim, tf_depth=3, dropout=0.15):
        super(SetTransformer, self).__init__()

        self.num_feats = num_feats
        self.k = num_output_points

        def dup(x):
            return (x, x) if type(x) == int else x

        self.lstm_layers = lstm_layers
        self.n_layers = dup(n_layers)
        self.n_heads = dup(n_heads)
        self.hidden_dim = dup(hidden_dim)
        self.ff_dim = dup(ff_dim)
        self.tf_depth = dup(tf_depth)

        self.d_model = [self.hidden_dim[i] * self.n_heads[i] for i in [0, 1]]

        encoder_builder_pre = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layers[0],
            n_heads=self.n_heads[0],
            query_dimensions=self.hidden_dim[0],
            value_dimensions=self.hidden_dim[0],
            feed_forward_dimensions=self.ff_dim[0],
            attention_type='linear',
            dropout=dropout
        )

        encoder_builder_post = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layers[1],
            n_heads=self.n_heads[1],
            query_dimensions=self.hidden_dim[1],
            value_dimensions=self.hidden_dim[1],
            feed_forward_dimensions=self.ff_dim[1],
            attention_type='linear',
            dropout=dropout
        )

        self.seeds = nn.Parameter(torch.normal(0, 1, (self.k, self.d_model[0])))
        self.encoder_pre = encoder_builder_pre.get()
        self.encoder_post = encoder_builder_post.get()

        self.initial_ff = nn.Linear(self.num_feats, self.d_model[0])
        # self.pos_encoding = PositionalEncoding(self.d_model[0], dropout=dropout)
        self.lstm = nn.LSTM(self.d_model[0], self.d_model[0], 2, batch_first=True, bidirectional=False)
        self.attn_pooling = AttentionLayer(LinearAttention(self.d_model[0]), self.d_model[0], self.n_heads[0])
        self.final_ff = nn.Linear(self.d_model[1], self.num_feats)

        # init masks to meaningless values, doesn't matter what. these are all empty anyway.
        self.mask = FullMask(N=self.k, M=5)
        self.kl_mask = LengthMask(torch.ones(5) * 5)
        self.ql_mask = LengthMask(torch.ones(5) * self.k)

    def forward(self, src, src_len_mask=None):

        x = self.initial_ff(src)

        # x = self.pos_encoding(x)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        for i in range(self.tf_depth[0]):
            x = self.encoder_pre(x)

        # in case the input sequence length has changed
        set_size = src.shape[1]
        batch_size = src.shape[0]
        if self.mask.shape[1] != set_size:
            self.mask = FullMask(N=self.k, M=set_size, device=src.device)
            self.kl_mask = LengthMask(torch.ones(batch_size) * set_size, device=src.device)
        # in case the batch size has changed
        if (self.ql_mask.shape[0] != batch_size) or (self.kl_mask.shape[0] != batch_size):
            self.ql_mask = LengthMask(torch.ones(batch_size) * self.k, device=src.device)
            self.kl_mask = LengthMask(torch.ones(batch_size) * set_size, device=src.device)

        # extend seeds to size of batch
        S = self.seeds.unsqueeze(0).repeat(batch_size, 1, 1)

        # perform pooling by multihead attention, reducing dimensionality of set
        x = self.attn_pooling(S, x, x, self.mask, self.ql_mask, self.kl_mask)

        for i in range(self.tf_depth[1]):
            x = self.encoder_post(x)

        x = self.final_ff(x)

        return x


if __name__ == '__main__':

    import model_params
    from chamferdist import ChamferDistance

    params = model_params.Params(r'params_default.json', False, 0)

    batch_size = 10
    seq_len = 100
    output_pts = 8
    num_feats = 2

    X = torch.rand(batch_size, seq_len, num_feats)
    tgt = torch.rand(batch_size, output_pts, num_feats)

    # model = SetTransformer(
    #     num_feats=num_feats,
    #     num_output_points=32,
    #     n_layers=1,
    #     tf_depth=2,
    #     n_heads=2,
    #     hidden_dim=64,
    #     ff_dim=64)

    # res = model(X)

    model = SetTransformer(**params.set_transformer_settings)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = ChamferDistance()

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
