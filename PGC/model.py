from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 400
SOS_token= '<SOS>'
EOS_token = '<EOS>'

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



class AttnPGCDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attention_vec_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnPGCDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention_vec_size = attention_vec_size


        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.atn_weight_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.atn_weight_dec_state = nn.Linear(self.hidden_size, self.hidden_size)

        self.encoder_atn_features = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, attention_vec_size), bias=True)
        self.coverage_atn_features = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, attention_vec_size), bias=True)


        #self.gen_

    def forward(self, input, hidden, encoder_states):
        decoder_state = self.embedding(input).view(1, 1, -1)
        decoder_state = self.dropout(decoder_state)


        '''
        attn_dist = [torch.tanh(self.atn_weight_hidden(enc_output) + self.atn_weight_dec_state())
                     for enc_output in encoder_outputs ]

        @ tensorflow:
        attention_vec_size = attn_size = encoder_states.get_shape()[2].value
        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable("v", [attention_vec_size])
        if use_coverage:
            with variable_scope.variable_scope("coverage"):
            w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])




        '''

        attention_vec_size = hidden.size()[0] # Unsure about this
             # W_h * h_i




        attn_weights = F.softmax(self.attn(torch.cat((decoder_state[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_states.unsqueeze(0))

        #p_generator =

        '''
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        '''
        return hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


teacher_forcing_ratio = 0.5
'''
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]


    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length
    '''
encoder = EncoderRNN(input_size=50, hidden_size=50, n_layers=1)
encoder_hidden = encoder.initHidden()

dummy_emb_matrix = torch.randn(10, 50)

decoder = AttnPGCDecoderRNN(hidden_size=50, output_size=50, attention_vec_size=60)

learning_rate = 0.15
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

input_length = input_variable.size()[0]
target_length = target_variable.size()[0]

encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

loss = 0

for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing












