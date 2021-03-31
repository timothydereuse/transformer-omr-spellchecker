import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.attention.attention_layer import AttentionLayer
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
import time


class SetTransformer(nn.Module):

    def __init__(self, num_feats, num_output_points, n_layers_prepooling, n_layers_postpooling,
                 n_heads, hidden_dim, ff_dim, tf_depth=3, dropout=0.15):
        super(SetTransformer, self).__init__()

        self.num_feats = num_feats
        self.k = num_output_points

        self.n_layers_prepooling = n_layers_prepooling
        self.n_layers_postpooling = n_layers_postpooling
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.d_model = hidden_dim * n_heads
        self.tf_depth = tf_depth

        encoder_builder_pre = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers_prepooling,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            attention_type='linear',
            dropout=dropout
        )

        encoder_builder_post = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers_postpooling,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            attention_type='linear',
            dropout=dropout
        )

        self.seeds = nn.Parameter(torch.normal(0, 1, (self.k, self.d_model)))
        self.encoder_pre = encoder_builder_pre.get()
        self.encoder_post = encoder_builder_post.get()

        self.attn_pooling = AttentionLayer(LinearAttention(self.d_model), self.d_model, self.n_heads)

        self.initial_ff = nn.Linear(self.num_feats, self.d_model)
        self.final_ff = nn.Linear(self.d_model, self.num_feats)

        # init masks to meaningless values, doesn't matter what. these are all empty anyway.
        self.mask = FullMask(N=self.k, M=5)
        self.kl_mask = LengthMask(torch.ones(5) * 5)
        self.ql_mask = LengthMask(torch.ones(5) * self.k)

    def forward(self, src, src_len_mask=None):

        x = self.initial_ff(src)
        # for _ in range(self.tf_depth):
        x = self.encoder_pre(x)

        # in case the input sequence length has changed
        set_size = src.shape[1]
        batch_size = src.shape[0]
        if self.mask.shape[1] != set_size:
            self.mask = FullMask(N=self.k, M=set_size)
            self.kl_mask = LengthMask(torch.ones(batch_size) * set_size)
        # in case the batch size has changed
        if (self.ql_mask.shape[0] != batch_size) or (self.kl_mask.shape[0] != batch_size):
            self.ql_mask = LengthMask(torch.ones(batch_size) * self.k)
            self.kl_mask = LengthMask(torch.ones(batch_size) * set_size)

        # extend seeds to size of batch
        S = self.seeds.unsqueeze(0).repeat(batch_size, 1, 1)

        # perform pooling by multihead attention, reducing dimensionality of set
        x = self.attn_pooling(S, x, x, self.mask, self.ql_mask, self.kl_mask)

        x = self.encoder_post(x)
        x = self.final_ff(x)

        return x


if __name__ == '__main__':

    import model_params as params
    from chamferdist import ChamferDistance

    batch_size = 10
    seq_len = 100
    output_pts = 8
    num_feats = 4

    X = torch.rand(batch_size, seq_len, num_feats)
    tgt = torch.rand(batch_size, output_pts, num_feats)

    model = SetTransformer(
        num_feats=num_feats,
        num_output_points=32,
        n_layers_prepooling=2,
        n_layers_postpooling=2,
        n_heads=2,
        hidden_dim=64,
        ff_dim=64)

    res = model(X)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = ChamferDistance()

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
