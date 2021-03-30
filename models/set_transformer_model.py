import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.attention.attention_layer import AttentionLayer
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
import fast_transformers
import time


class SetTransformer(nn.Module):

    def __init__(self, num_feats, num_output_points, n_layers, n_heads, hidden_dim, ff_dim, tf_depth=3, dropout=0.15):
        super(SetTransformer, self).__init__()

        self.input_feats = num_feats
        self.output_feats = num_feats
        self.k = num_output_points

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.d_model = hidden_dim * n_heads
        self.tf_depth = tf_depth

        encoder_builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            attention_type='linear',
            dropout=dropout
        )

        self.seeds = nn.Parameter(torch.rand(self.k, self.d_model))
        self.encoder = encoder_builder.get()

        self.attn_pooling = AttentionLayer(LinearAttention(self.d_model), self.d_model, self.n_heads)

        self.src_embed = nn.Linear(self.input_feats, self.d_model)
        self.final_ff = nn.Linear(self.d_model, self.output_feats)

        # init masks to meaningless values, doesn't matter what. these are all empty anyway.
        self.mask = FullMask(N=self.k, M=5)
        self.kl_mask = LengthMask(torch.ones(5) * 5)
        self.ql_mask = LengthMask(torch.ones(5) * self.k)

    def forward(self, src, src_len_mask=None):

        x = src
        # for _ in range(self.tf_depth):
        x = self.encoder(x)

        # in case the input sequence length has changed
        set_size = src.shape[1]
        batch_size = src.shape[0]
        if self.mask.shape[1] != set_size:
            self.mask = FullMask(N=self.k, M=set_size)
            self.kl_mask = LengthMask(torch.ones(batch_size) * set_size)
        # in case the batch size has changed
        if (self.ql_mask.shape[0] != batch_size) or (self.kl_mask.shape[0] != batch_size):
            self.ql_mask = LengthMask(torch.ones(batch_size) * k)
            self.kl_mask = LengthMask(torch.ones(batch_size) * set_size)

        # extend seeds to size of batch
        S = self.seeds.unsqueeze(0).repeat(self.shape[0], 1, 1)

        # perform pooling by multihead attention, reducing dimensionality of set
        x = self.attn_pooling(S, x, x, self.mask, self.ql_mask, self.kl_mask)

        return x


if __name__ == '__main__':

    import model_params as params

    batch_size = params.batch_size
    seq_len = params.seq_length
    tgt_seq_len = 7

    input_feats = params.transformer_ar_settings['input_feats']
    output_feats = params.transformer_ar_settings['output_feats']
    # n_layers = 2
    # n_heads = 1
    # hidden_dim = 64
    # ff_dim = 64

    X = torch.rand(batch_size, seq_len, input_feats)
    tgt = torch.rand(batch_size, tgt_seq_len, output_feats)

    x_mask = fast_transformers.masking.TriangularCausalMask(tgt_seq_len)
    lengths = torch.randint(seq_len, (batch_size,))
    src_len_mask = fast_transformers.masking.LengthMask(lengths, max_len=seq_len)
    lengths = torch.randint(tgt_seq_len, (batch_size,))
    tgt_len_mask = fast_transformers.masking.LengthMask(lengths, max_len=tgt_seq_len)

    # query_dimensions = hidden_dim
    # value_dimensions = hidden_dim
    # attention_type = 'linear'
    #
    # d_model = value_dimensions * n_heads

    # model = TransformerEncoderDecoder(input_feats, output_feats, n_layers, n_heads, hidden_dim, ff_dim)
    model = TransformerEncoderDecoder(**params.transformer_ar_settings)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss(reduction='mean')

    num_epochs = 10

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    model.train()
    epoch_loss = 0
    for i in range(num_epochs):

        start_time = time.time()
        optimizer.zero_grad()

        output = model(X, tgt, src_len_mask=src_len_mask, tgt_len_mask=tgt_len_mask)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

    model.eval()
    res = model.inference_decode(X, tgt_seq_len)


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
