from __future__ import unicode_literals, print_function, division

import random
import time
import math
import Pytorch_seq2seq.data_prepare as data_prepare
import Pytorch_seq2seq.seq2seq_model as models

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

input_lang, output_lang, pairs = data_prepare.prepareData(data_prepare.load_dailyMail())
print(random.choice(pairs))

hidden_size = 256
encoder = models.EncoderRNN(input_lang.n_words, hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size,
                 n_layers=1, dropout_p=0.1, encoder_max_length=100, decoder_max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

        self.embedding = nn.Embedding(self.embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)

        # Attention variables
        self.w_h = Variable(torch.randn(self.hidden_size))
        self.w_s = Variable(torch.randn(self.hidden_size))
        self.att_bias = Variable(torch.randn(1))
        self.attn_weight_v = Variable(torch.randn(self.hidden_size))

        # Generator
        self.gen_layer = nn.Linear(self.hidden_size*3 + self.embedding_size, 1)
        self.out_hidden = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out_vocab = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_token, last_decoder_hidden, encoder_states):

        embedded_input = self.embedding(input_token).view(1, 1, -1)
        embedded_input = self.dropout(embedded_input)
        decoder_output, decoder_hidden = self.gru(embedded_input, last_decoder_hidden)

        att_dist = F.tanh((self.w_h * encoder_states) + (self.w_s * decoder_output) + self.att_bias)
        att_dist = (self.attn_weight_v * att_dist).sum(-1)
        att_dist = F.softmax(att_dist, dim=-1)
        context_vector = (att_dist.view(1, -1, 1) * encoder_states).sum(1)

        dec_out_and_context = torch.cat((decoder_output.view(1, -1), context_vector), 1)
        p_gen = F.sigmoid(self.gen_layer(torch.cat(dec_out_and_context, embedded_input), 1))
        p_vocab = F.softmax(self.out_vocab(self.out_hidden(torch.cat(dec_out_and_context)))) # replace with embedding weight

        return decoder_hidden, p_gen, p_vocab, att_dist

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result


decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
training_pairs = [data_prepare.variablesFromPair(pairs[0], input_lang, output_lang)]
criterion = nn.NLLLoss()

training_pair = training_pairs[0]
input_variable = training_pair[0]
target_variable = training_pair[1]

teacher_forcing_ratio = 0.5
encoder_hidden = encoder.initHidden()

encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

input_length = input_variable.size()[0]
target_length = target_variable.size()[0]

encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

loss = 0

for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

SOS_token = 0
EOS_token = 1

decoder_input = Variable(torch.LongTensor([[SOS_token]]))

decoder_hidden = encoder_hidden

# Teacher forcing: Feed the target as the next input
for di in range(target_length):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
    loss += criterion(decoder_output, target_variable[di])
    decoder_input = target_variable[di]  # Teacher forcing

loss.backward()

encoder_optimizer.step()
decoder_optimizer.step()

print(loss.data[0])

