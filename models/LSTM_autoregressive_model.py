import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=self.num_layers, bidirectional=True, batch_first=True)

    def forward(self, input):
        embedded = torch.relu(self.embedding(input))
        output = embedded
        output, hidden = self.lstm(output)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=3, dropout_p=0.1, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, targets, hidden, encoder_outputs):
        embedded = torch.relu(self.embedding(targets))
        embedded = self.dropout(embedded)

        attn_input = torch.cat((embedded, hidden[0]), 1)
        attn_weights = torch.softmax(self.attn(attn_input), dim=1)
        print(attn_weights.shape, encoder_outputs[0].shape)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((embedded[0], attn_applied[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)
        output = torch.relu(output)

        output, hidden = self.lstm(output, hidden)
        return output, hidden, attn_weights


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
