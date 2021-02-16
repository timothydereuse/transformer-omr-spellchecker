import torch
from fast_transformers.builders import TransformerEncoderBuilder

# Create the builder for our transformers
builder = TransformerEncoderBuilder.from_kwargs(
    n_layers=8,
    n_heads=8,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=1024
)

# Build a transformer with linear attention
builder.attention_type = "linear"
linear_model = builder.get()

# Construct the dummy input
X = torch.rand(10, 40, 8*64)


linear_model.eval()

with torch.no_grad():
    y = linear_model(X)
