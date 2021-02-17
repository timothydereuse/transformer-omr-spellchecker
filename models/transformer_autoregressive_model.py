import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
import fast_transformers

batch_size = 16
seq_len = 20
tgt_seq_len = 7


input_feats = 4
output_feats = 3
n_layers = 4
n_heads = 4
hidden_dim = 32
ff_dim = 128

query_dimensions = hidden_dim
value_dimensions = hidden_dim
attention_type = 'linear'

d_model = value_dimensions * n_heads


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, input_feats, output_feats, n_layers, n_heads, hidden_dim, ff_dim):
        super(EncoderDecoder, self).__init__()

        self.input_feats = input_feats
        self.output_feats = output_feats

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.d_model = value_dimensions * n_heads

        encoder_builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            attention_type='linear'
        )

        decoder_builder = TransformerDecoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            cross_attention_type='linear',
            self_attention_type='causal-linear'
        )

        self.encoder = encoder_builder.get()
        self.decoder = decoder_builder.get()
        self.src_embed = nn.Linear(self.input_feats, self.d_model)
        self.tgt_embed = nn.Linear(self.output_feats, self.d_model)
        self.final_ff = nn.Linear(self.d_model, self.output_feats)

    def forward(self, src, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        decoded = self.decode(tgt, self.encode(src), tgt_mask)
        decoded = self.final_ff(decoded)
        return decoded

    def encode(self, src):
        src_embedded = self.src_embed(src)
        return self.encoder(src_embedded)

    def decode(self, tgt, memory, tgt_mask):
        tgt_embedded = self.tgt_embed(tgt)
        return self.decoder(tgt_embedded, memory, tgt_mask)

#
#
# # Build a transformer with linear attention builder.
# encoder = encoder_builder.get()
# decoder = decoder_builder.get()

# Construct the dummy input
X = torch.rand(batch_size, seq_len, input_feats)
tgt = torch.rand(batch_size, tgt_seq_len, output_feats)

x_mask = fast_transformers.masking.TriangularCausalMask(tgt_seq_len)

model = EncoderDecoder(input_feats, output_feats, n_layers, n_heads, hidden_dim, ff_dim)


# with torch.no_grad():
#     memory = encoder(X)
#     y_pred = decoder(tgt, memory, x_mask)
