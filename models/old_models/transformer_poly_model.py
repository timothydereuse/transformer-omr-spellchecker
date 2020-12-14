import math
import numpy as np
import torch
import torch.nn as nn


class TransformerPolyphonic(nn.Module):

    def __init__(self, num_feats, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerPolyphonic, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.num_feats = num_feats
        self.ninp = ninp
        self.nhid = nhid

        self.encoder = self.decoder = nn.Sequential(
            nn.Linear(num_feats, self.ninp),
            nn.LeakyReLU(),
            nn.Linear(self.ninp, self.ninp)
            )

        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        encoder_layers = TransformerEncoderLayer(self.ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Sequential(
            nn.Linear(self.ninp, self.num_feats),
            nn.ReLU(),
            nn.Linear(self.num_feats, self.num_feats),
            )

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder[0].bias.data.zero_()

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.num_feats)
        src = self.pos_encoder(src)
        trans_output = self.transformer_encoder(src, self.src_mask)
        decode_output = self.decoder(trans_output)

        return decode_output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bptt = 2s0
    batch_size = 20

    num_feats = 64
    emsize = 100    # embedding dimension
    nhid = 100      # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 3     # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2       # the number of heads in the multiheadattention models
    dropout = 0.2   # the dropout value
    model = TransformerPolyphonic(num_feats, emsize, nhead, nhid, nlayers, dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)

    full_dat = torch.rand(bptt + 1, batch_size, num_feats).round().float()
    # full_dat = torch.floor(torch.rand(bptt + 1, batch_size) * num_feats).long()
    data = full_dat[:-1].to(device)
    # embedded_data = torch.zeros(data.shape + (num_feats,))
    # for i in range(bptt):
    #     for j in range(batch_size):
    #         embedded_data[i, j, data[i, j]] = 1
    targets = full_dat[1:].to(device)

    criterion = nn.BCEWithLogitsLoss()
    lr = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.99)

    model.train()
    total_loss = 0.
    start_time = time.time()
    num_batches = 1000
    for batch in range(num_batches):
        # data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, num_feats), targets.view(-1, num_feats))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    1, batch, num_batches, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()
        scheduler.step()
