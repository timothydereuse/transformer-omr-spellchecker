import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
import fast_transformers
import time

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


class TransformerEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, input_feats, output_feats, n_layers, n_heads, hidden_dim, ff_dim, dropout=0.15):
        super(TransformerEncoderDecoder, self).__init__()

        self.input_feats = input_feats
        self.output_feats = output_feats

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.d_model = value_dimensions * n_heads

        self.A = nn.GELU()

        encoder_builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            attention_type='linear',
            dropout=dropout
        )

        decoder_builder = TransformerDecoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=hidden_dim,
            value_dimensions=hidden_dim,
            feed_forward_dimensions=ff_dim,
            cross_attention_type='linear',
            self_attention_type='causal-linear',
            dropout=dropout
        )

        self.encoder = encoder_builder.get()
        self.decoder = decoder_builder.get()
        self.src_embed = nn.Linear(self.input_feats, self.d_model)
        self.tgt_embed = nn.Linear(self.output_feats, self.d_model)
        self.final_ff = nn.Linear(self.d_model, self.output_feats)

    def forward(self, src, tgt, tgt_mask, src_len_mask=None, tgt_len_mask=None):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, len_mask=src_len_mask)
        decoded = self.decode(tgt, memory, tgt_mask, len_mask=tgt_len_mask)
        decoded = self.A(self.final_ff(decoded))
        return decoded

    def encode(self, src, len_mask=None):
        src_embedded = self.A(self.src_embed(src))
        return self.encoder(src_embedded, length_mask=len_mask)

    def decode(self, tgt, memory, tgt_mask, len_mask=None):
        tgt_embedded = self.A(self.tgt_embed(tgt))
        return self.decoder(tgt_embedded, memory, tgt_mask, x_length_mask=len_mask)

if __name__ == '__main__':
    X = torch.rand(batch_size, seq_len, input_feats)
    tgt = torch.rand(batch_size, tgt_seq_len, output_feats)

    x_mask = fast_transformers.masking.TriangularCausalMask(tgt_seq_len)
    lengths = torch.randint(seq_len, (batch_size,))
    src_len_mask = fast_transformers.masking.LengthMask(lengths, max_len=seq_len)
    lengths = torch.randint(tgt_seq_len, (batch_size,))
    tgt_len_mask = fast_transformers.masking.LengthMask(lengths, max_len=tgt_seq_len)

    model = TransformerEncoderDecoder(input_feats, output_feats, n_layers, n_heads, hidden_dim, ff_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss(reduction='mean')

    num_epochs = 100

    n_params = sum(p.numel() for p in model.parameters())
    print(f'created model with n_params={n_params}')

    model.train()
    epoch_loss = 0
    for i in range(num_epochs):

        start_time = time.time()
        optimizer.zero_grad()

        output = model(X, tgt, tgt_mask=x_mask, src_len_mask=src_len_mask, tgt_len_mask=tgt_len_mask)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        elapsed = time.time() - start_time
        print(f"epoch: {i} | loss: {loss.item():2.5f} | time: {elapsed:2.5f}")

    # with torch.no_grad():
    #     memory = encoder(X)
    #     y_pred = decoder(tgt, memory, x_mask)
