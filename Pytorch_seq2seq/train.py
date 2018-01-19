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

#input_lang, output_lang, pairs = data_prepare.prepareData(data_prepare.load_dailyMail())
#print(random.choice(pairs))



class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        return self.gru(embedded, hidden)

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size,
                 n_layers=1, dropout_p=0.1, encoder_max_length=100, decoder_max_length=100):
        super(AttnDecoderRNN, self).__init__()
        # Parameters
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)

        # Attention variables
        self.w_h = Variable(torch.randn(self.hidden_size))
        self.w_s = Variable(torch.randn(self.hidden_size))
        self.att_bias = Variable(torch.randn(1))
        self.attn_weight_v = Variable(torch.randn(self.hidden_size))

        # Generator
        self.gen_layer = nn.Linear(self.hidden_size*2 + self.embedding_size, 1)
        self.out_hidden = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out_vocab = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, input_token, last_decoder_hidden, encoder_states):
        embedded_input= self.embedding(input_token)
        embedded_input = self.dropout(embedded_input)
        decoder_output, decoder_hidden = self.gru(embedded_input, torch.unsqueeze(last_decoder_hidden, 0))

        att_dist = F.tanh((self.w_h * encoder_states) + (self.w_s * decoder_output) + self.att_bias)
        att_dist = (self.attn_weight_v * att_dist).sum(-1)
        att_dist = F.softmax(att_dist, dim=-1)
        #print('att_dist', att_dist.size(), 'encoder_states', encoder_states.size(), 'att_dist-view', torch.unsqueeze(att_dist, 2))

        context_vector = (torch.unsqueeze(att_dist, 2) * encoder_states).sum(1)
        decoder_context = torch.cat((torch.squeeze(decoder_output, 1), context_vector), -1)
        p_vocab = F.softmax(self.out_vocab(self.out_hidden(decoder_context))) # replace with embedding weight

        p_gen = F.sigmoid(self.gen_layer(torch.cat((decoder_context, torch.squeeze(embedded_input, 1)), 1)))

        return decoder_hidden, p_gen, p_vocab, att_dist

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result


embedding_size = 128
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size=embedding_size)

decoder = AttnDecoderRNN(hidden_size, embedding_size, output_lang.n_words, 1, dropout_p=0.1)

learning_rate=0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


training_pairs = [data_prepare.variablesFromPair(pairs[i], input_lang, output_lang) for i in range(50)]
criterion = nn.NLLLoss()

batch_size = 32
input_batch = training_pairs[:batch_size]


input_length = 100
input_variable = Variable(torch.LongTensor(batch_size, input_length))
for b in range(batch_size):
    for i in range(len(input_batch[b][0])): input_variable[b, i] = input_batch[b][0][i]
    for i in range(input_length - len(input_batch[b][0])): input_variable[b, input_length-i-1] = 0



teacher_forcing_ratio = 0.5
encoder_hidden = encoder.initHidden()

encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

target_length = 30
input_length = 100
encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

loss = 0

encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

SOS_token = 1
EOS_token = 2

decoder_input = Variable(torch.LongTensor([[SOS_token] for i in range(batch_size)]))
decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)



# Teacher forcing: Feed the target as the next input
for di in range(target_length):
    decoder_hidden, p_gen, p_vocab, attention_dist = decoder(decoder_input, decoder_hidden, encoder_outputs)
    p_final = p_gen * p_vocab
    #decoder_output = target_variable[di] # perfrom beam search
    #loss += criterion(decoder_output, target_variable[di])
    #decoder_input = target_variable[di]  # Teacher forcing
    break

#loss.backward()

encoder_optimizer.step()
decoder_optimizer.step()

#print(loss.data[0])

