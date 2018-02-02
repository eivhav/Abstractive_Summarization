from __future__ import unicode_literals, print_function, division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
'''
use_cuda = False#torch.cuda.is_available()
multi_gpu = False
from Pytorch_seq2seq.data_loader import Vocab, TextPair, DataSet
data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/'

#if 'dataset' not in globals():
with open(data_path +'DM_25k_summary.pickle', 'rb') as f:
    dataset = pickle.load(f)
'''


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        return self.gru(embedded, hidden)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size,
                 n_layers=1, dropout_p=0.1, encoder_max_length=100, decoder_max_length=100, embedding_weight=None):
        super(AttnDecoderRNN, self).__init__()
        # Parameters
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
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

    def forward(self, input_token, prev_decoder_h_states, encoder_states, full_input_var, previous_att=None):
        embedded_input = self.embedding(input_token)
        embedded_input = self.dropout(embedded_input)

        last_hidden = prev_decoder_h_states.narrow(1, prev_decoder_h_states.size()[1]-1, 1).squeeze(1)
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

        p_vocab = self.out_vocab(self.out_hidden(combined_context)) # replace with embedding weight
        p_gen = F.sigmoid(self.gen_layer(combined_context))
        '''
        pointer_dist = att_dist * (1-p_gen)
        padding_matrix = Variable(torch.zeros(batch_size, 250)).cuda()
        generator_dist = torch.cat((p_vocab * p_gen, padding_matrix), 1)
        p_final = generator_dist.scatter_add_(1, full_input_var, pointer_dist)
        '''
        token_input_dist = Variable(torch.zeros((full_input_var.size()[0], self.vocab_size+250)))
        padding_matrix_2 = Variable(torch.zeros(full_input_var.size()[0], 250))
        if use_cuda:
            token_input_dist = token_input_dist.cuda()
            padding_matrix_2 = padding_matrix_2.cuda()

        token_input_dist.scatter_add_(1, full_input_var, att_dist)
        p_final = torch.cat((p_vocab * p_gen, padding_matrix_2), 1) + (1-p_gen) * token_input_dist

        decoder_h_states = torch.cat((prev_decoder_h_states, decoder_hidden.unsqueeze(1)), 1)
        return decoder_h_states, p_final, p_gen, p_vocab, att_dist, previous_att



    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result


SOS_token = 1
EOS_token = 2
UNK_token = 3

batch_size = 3
embedding_size = 128
hidden_size = 256
input_length = 300
target_length = 50
vocab_size = len(dataset.vocab.index2word)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.8)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

encoder = EncoderRNN(vocab_size, hidden_size=embedding_size)

emb_w = encoder.embedding.weight
decoder = AttnDecoderRNN(hidden_size, embedding_size, vocab_size, 1, dropout_p=0.1, embedding_weight=None)

if use_cuda:
    encoder.cuda()
    decoder.cuda()

learning_rate= 0.015
encoder_optimizer = optim.Adagrad(encoder.parameters(), lr= learning_rate, weight_decay=0.0000001)# lr=learning_rate)
decoder_optimizer = optim.Adagrad(decoder.parameters(), lr= learning_rate, weight_decay=0.0000001)
criterion = nn.NLLLoss()
teacher_forcing_ratio = 0.5


def zero_pad(tokens, len_limit):
    if len(tokens) < len_limit: return tokens + [0] * (len_limit - len(tokens))
    else: return tokens[:len_limit]



def train_batch(nb, samples):
    last = time.time()
    input_variable = Variable(torch.LongTensor([zero_pad(pair.masked_source_tokens, input_length) for pair in samples]))
    full_input_variable = Variable(torch.LongTensor([zero_pad(pair.full_source_tokens, input_length) for pair in samples]))
    target_variable = Variable(torch.LongTensor([zero_pad(pair.masked_target_tokens, target_length) for pair in samples]))
    full_target_variable = Variable(torch.LongTensor([zero_pad(pair.full_target_tokens, target_length) for pair in samples]))

    encoder_hidden = Variable(torch.zeros(2, batch_size, embedding_size))
    if use_cuda:
        input_variable = input_variable.cuda()
        full_input_variable = full_input_variable.cuda()
        target_variable = target_variable.cuda()
        full_target_variable = full_target_variable.cuda()
        encoder_hidden= encoder_hidden.cuda()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token] for i in range(batch_size)]))
    if use_cuda: decoder_input = decoder_input.cuda()

    decoder_hidden_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
    previous_att = None
    count = 0
    for token_i in range(target_length):
        print("iteration ", count)
        count += 1
        decoder_hidden_states, p_final, p_gen, p_vocab, attention_dist, previous_att = \
            decoder(decoder_input, decoder_hidden_states, encoder_outputs, full_input_variable, previous_att)
        loss += criterion(F.log_softmax(p_vocab, dim=1), target_variable.narrow(1, token_i, 1).squeeze(-1))
        decoder_input = target_variable.narrow(1, token_i, 1) # Teacher forcing

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


nb_epochs = 20
batch = training_pairs[:batch_size]
epoch_loss = 0
epoch_loss += train_batch(0, batch)











