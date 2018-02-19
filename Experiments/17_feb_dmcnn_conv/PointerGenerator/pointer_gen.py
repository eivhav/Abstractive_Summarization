from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = int(hidden_size/2)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, self.hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        return self.gru(embedded, hidden)

    def init_hidden(self, batch_size, use_cuda):
        if use_cuda: return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()
        else: return Variable(torch.zeros(2, batch_size, self.hidden_size))



class CoverageAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, n_layers=1, dropout_p=0.1,
                 input_lenght=400, embedding_weight=None, use_conv=False):
        super(CoverageAttnDecoderRNN, self).__init__()
        # Parameters
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.vocab_size = vocab_size
        self.use_conv = use_conv
        if use_conv: print('use_conv')

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)

        # Attention variables
        self.w_h_conv = nn.Conv2d(self.hidden_size, self.hidden_size, 1, stride=1)
        self.w_h = nn.Parameter(torch.randn(self.hidden_size))
        self.w_s = nn.Parameter(torch.randn(self.hidden_size))
        self.w_c = nn.Parameter(torch.randn(input_lenght))
        self.att_bias = nn.Parameter(torch.randn(1))
        self.attn_weight_v = nn.Parameter(torch.randn(self.hidden_size))

        # Generator
        self.gen_layer = nn.Linear(self.hidden_size*2 + self.embedding_size, 1)
        self.out_hidden = nn.Linear(self.hidden_size*2, self.embedding_size)
        self.out_vocab = nn.Linear(self.embedding_size, vocab_size)

        # Weight_sharing
        if embedding_weight is not None:
            self.out_vocab.weight = embedding_weight

    def forward(self, input_token, prev_decoder_h_states, last_hidden, encoder_states, full_input_var,
                previous_att, nb_unk_tokens, use_cuda):

        embedded_input = self.embedding(input_token)
        embedded_input = self.dropout(embedded_input)
        decoder_output, decoder_hidden = self.gru(embedded_input, torch.unsqueeze(last_hidden, 0))
        decoder_hidden = decoder_hidden.squeeze(0)

        if self.use_conv:
            encoder_features = self.w_h_conv(torch.transpose(encoder_states, 1, 2).unsqueeze(-1))
            encoder_features = torch.transpose(encoder_features, 2, 1).squeeze(-1)
        else:
            encoder_features = self.w_h * encoder_states

        if previous_att is None:
            att_dist = F.tanh(encoder_features + (self.w_s * decoder_output) + self.att_bias)
            att_dist = (self.attn_weight_v * att_dist).sum(-1)
            att_dist = F.softmax(att_dist, dim=-1)
            previous_att = att_dist.unsqueeze(1)
        else:
            att_dist = F.tanh(encoder_features + (self.w_s * decoder_output) +
                              (self.w_c * previous_att.sum(1)).unsqueeze(-1) + self.att_bias)
            att_dist = (self.attn_weight_v * att_dist).sum(-1)
            att_dist = F.softmax(att_dist, dim=-1)
            previous_att = torch.cat((previous_att, att_dist.unsqueeze(1)), 1)

        encoder_context = (torch.unsqueeze(att_dist, 2) * encoder_states).sum(1)
        combined_context = torch.cat((torch.squeeze(decoder_output, 1), encoder_context), -1)

        p_vocab = F.softmax(self.out_vocab(self.out_hidden(combined_context)), dim=-1)
        p_gen = F.sigmoid(self.gen_layer(torch.cat((combined_context, torch.squeeze(embedded_input, 1)), 1)))

        token_input_dist = Variable(torch.zeros((full_input_var.size()[0], self.vocab_size+nb_unk_tokens+200)))
        padding_matrix_2 = Variable(torch.zeros(full_input_var.size()[0], nb_unk_tokens+200))
        if use_cuda:
            token_input_dist = token_input_dist.cuda()
            padding_matrix_2 = padding_matrix_2.cuda()

        token_input_dist.scatter_add_(1, full_input_var, att_dist)
        p_final = torch.cat((p_vocab * p_gen, padding_matrix_2), 1) + (1-p_gen) * token_input_dist

        return p_final, p_gen, p_vocab, att_dist, prev_decoder_h_states, decoder_hidden, previous_att

    def init_hidden(self, use_cuda):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result



class TemporalAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, n_layers=1, dropout_p=0.1,  embedding_weight=None):
        super(TemporalAttnDecoderRNN, self).__init__()
        # Parameters
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)

        # Attention variables
        self.w_h = nn.Parameter(torch.randn(self.hidden_size))
        self.w_d = nn.Parameter(torch.randn(self.hidden_size))

        # Generator
        self.out_hidden = nn.Linear(self.hidden_size*3, self.embedding_size)
        self.out_vocab = nn.Linear(self.embedding_size, vocab_size)
        self.gen_layer = nn.Linear(self.hidden_size*3, 1)

        # Weight_sharing
        if embedding_weight is not None:
            self.out_vocab.weight = embedding_weight

    def forward(self, input_token, prev_decoder_h_states, last_hidden, encoder_states,
                full_input_var, previous_att, nb_unk_tokens, use_cuda):
        embedded_input = self.embedding(input_token)
        embedded_input = self.dropout(embedded_input)
        decoder_output, decoder_hidden = self.gru(embedded_input, torch.unsqueeze(last_hidden, 0))
        decoder_hidden = decoder_hidden.squeeze(0)

        att_dist = (decoder_hidden.unsqueeze(1) * (self.w_h * encoder_states)).sum(-1)

        if previous_att is None:
            temporal_att = torch.exp(att_dist)
            previous_att = att_dist.unsqueeze(1)
            att_dist = temporal_att / temporal_att.sum(-1).unsqueeze(-1)
            encoder_context = (torch.unsqueeze(att_dist, 2) * encoder_states).sum(1)

            decoder_context = Variable(torch.zeros(input_token.size()[0], self.hidden_size))
            if use_cuda: decoder_context = decoder_context.cuda()
        else:
            temporal_att = torch.exp(att_dist) / torch.exp(previous_att).sum(1)
            previous_att = torch.cat((previous_att, att_dist.unsqueeze(1)), 1)
            att_dist = temporal_att / temporal_att.sum(-1).unsqueeze(-1)
            encoder_context = (torch.unsqueeze(att_dist, 2) * encoder_states).sum(1)

            decoder_att_dist = (decoder_hidden.unsqueeze(1) * (self.w_d * prev_decoder_h_states)).sum(-1)
            decoder_att_dist = F.softmax(decoder_att_dist, -1)
            decoder_context = (torch.unsqueeze(decoder_att_dist, 2) * prev_decoder_h_states).sum(1)

        combined_context = torch.cat((torch.cat((decoder_hidden, encoder_context), -1), decoder_context), -1)

        p_vocab = F.softmax(self.out_vocab(self.out_hidden(combined_context)), dim=-1) # replace with embedding weight
        p_gen = F.sigmoid(self.gen_layer(combined_context))

        token_input_dist = Variable(torch.zeros((full_input_var.size()[0], self.vocab_size+nb_unk_tokens+200)))
        padding_matrix_2 = Variable(torch.zeros(full_input_var.size()[0], nb_unk_tokens+200))
        if use_cuda:
            token_input_dist = token_input_dist.cuda()
            padding_matrix_2 = padding_matrix_2.cuda()

        token_input_dist.scatter_add_(1, full_input_var, att_dist)
        p_final = torch.cat((p_vocab * p_gen, padding_matrix_2), 1) + (1-p_gen) * token_input_dist

        decoder_h_states = torch.cat((prev_decoder_h_states, decoder_hidden.unsqueeze(1)), 1)
        return p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att

    def init_hidden(self, use_cuda):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result




