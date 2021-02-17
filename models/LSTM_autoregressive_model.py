import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_feats, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = nn.Linear(input_feats, emb_dim)  # no dropout as only one layer!
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)  # no cell state!

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim = 2)
        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim = 1)
        # output = [batch size, emb dim + hid dim * 2]
        prediction = self.fc_out(output)
        # prediction = [batch size, output dim]
        return prediction, hidden

teacher_forcing_ratio = 0.5

hidden_size = 128
lr = 0.01
inp_feats = 4
batch_size = 100
seq_len = 50
output_feats = 3
num_layers = 3

encoder = EncoderRNN(inp_feats, hidden_size, num_layers)
attn_decoder = AttnDecoderRNN(hidden_size, output_feats, num_layers, dropout_p=0.1)
encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
decoder_optimizer = torch.optim.SGD(attn_decoder.parameters(), lr=lr)

inp = torch.rand(batch_size, seq_len, inp_feats)
encoder_outputs, encoder_hidden = encoder(inp)

# necessary to do this reshape to put the batch dimension back in position 0.
decoder_hidden = (
    torch.cat((encoder_hidden[0][:num_layers], encoder_hidden[0][num_layers:]), 2).transpose(0, 1),
    torch.cat((encoder_hidden[0][:num_layers], encoder_hidden[0][num_layers:]), 2).transpose(0, 1)
)

targets = torch.zeros(batch_size, 1, output_feats)

x = attn_decoder(targets, decoder_hidden, encoder_outputs)

# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
#
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#
#         assert encoder.hid_dim == decoder.hid_dim, \
#             "Hidden dimensions of encoder and decoder must be equal!"
#
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#
#         # src = [src len, batch size]
#         # trg = [trg len, batch size]
#         # teacher_forcing_ratio is probability to use teacher forcing
#         # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
#
#         batch_size = trg.shape[1]
#         trg_len = trg.shape[0]
#         trg_vocab_size = self.decoder.output_dim
#
#         #tensor to store decoder outputs
#         outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
#
#         #last hidden state of the encoder is the context
#         context = self.encoder(src)
#
#         #context also used as the initial hidden state of the decoder
#         hidden = context
#
#         #first input to the decoder is the <sos> tokens
#         input = trg[0,:]
#
#         for t in range(1, trg_len):
#
#             #insert input token embedding, previous hidden state and the context state
#             #receive output tensor (predictions) and new hidden state
#             output, hidden = self.decoder(input, hidden, context)
#
#             #place predictions in a tensor holding predictions for each token
#             outputs[t] = output
#
#             #decide if we are going to use teacher forcing or not
#             teacher_force = random.random() < teacher_forcing_ratio
#
#             #get the highest predicted token from our predictions
#             top1 = output.argmax(1)
#
#             #if teacher forcing, use actual next token as next input
#             #if not, use predicted token
#             input = trg[t] if teacher_force else top1
#
#         return outputs
#
#
#
#
#
# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=300):
#     encoder_hidden = encoder.initHidden()
#
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)
#
#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
#
#     loss = 0
#
#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(
#             input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]
#
#     decoder_input = torch.tensor([[SOS_token]], device=device)
#
#     decoder_hidden = encoder_hidden
#
#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
#
#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             loss += criterion(decoder_output, target_tensor[di])
#             decoder_input = target_tensor[di]  # Teacher forcing
#
#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input
#
#             loss += criterion(decoder_output, target_tensor[di])
#             if decoder_input.item() == EOS_token:
#                 break
#
#     loss.backward()
#
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.item() / target_length
