from __future__ import unicode_literals, print_function, division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle

use_cuda = torch.cuda.is_available()
#from Pytorch_seq2seq.data_loader import *
data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/'

#if 'dataset' not in globals():
with open(path_2 +'DM_25k_summary.pickle', 'rb') as f:
    dataset = pickle.load(f)



class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        return self.gru(embedded, hidden)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


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

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)

        # Attention variables
        self.w_h = nn.Parameter(torch.randn(self.hidden_size)).requires_grad
        self.w_s = nn.Parameter(torch.randn(self.hidden_size))
        self.att_bias = nn.Parameter(torch.randn(1))
        self.attn_weight_v = nn.Parameter(torch.randn(self.hidden_size))

        # Generator
        self.gen_layer = nn.Linear(self.hidden_size*2 + self.embedding_size, 1)
        self.out_hidden = nn.Linear(self.hidden_size*2, self.embedding_size)
        self.out_vocab = nn.Linear(self.embedding_size, vocab_size)

        # Weight_sharing
        if embedding_weight is not None:
            self.out_vocab.weight = embedding_weight


    def forward(self, input_token, last_decoder_hidden, encoder_states):
        embedded_input = self.embedding(input_token)
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

        return decoder_hidden.squeeze(0), p_gen, p_vocab, att_dist

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result


SOS_token = 1
EOS_token = 2
UNK_token = 3

batch_size = 64
embedding_size = 128
hidden_size = 256
input_length = 400
target_length = 60
vocab_size = len(dataset.vocab.index2word)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.8)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

encoder = EncoderRNN(vocab_size, hidden_size=embedding_size)
emb_w = encoder.embedding.weight
decoder = AttnDecoderRNN(hidden_size, embedding_size, vocab_size, 1, dropout_p=0.1, embedding_weight=emb_w)

if use_cuda:
    encoder.cuda()
    decoder.cuda()

#learning_rate=0.01
encoder_optimizer = optim.Adam(encoder.parameters())# lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters())
criterion = nn.NLLLoss()
teacher_forcing_ratio = 0.5


def zero_pad(tokens, len_limit):
    if len(tokens) < len_limit: return tokens + [0] * (len_limit - len(tokens))
    else: return tokens[:len_limit]




def train_batch(batch_nb):
    last = time.time()

    input_batch = [pair for pair in training_pairs[batch_size*batch_nb:batch_size*(batch_nb+1)]]
    target_batch = [pair for pair in training_pairs[batch_size*batch_nb:batch_size*(batch_nb+1)]]

    input_variable = Variable(torch.LongTensor([zero_pad(pair.masked_source_tokens, input_length) for pair in input_batch]))
    full_input_variable = Variable(torch.LongTensor([zero_pad(pair.full_source_tokens, input_length) for pair in input_batch]))
    target_variable = Variable(torch.LongTensor([zero_pad(pair.masked_target_tokens, target_length) for pair in target_batch]))
    full_target_variable = Variable(torch.LongTensor([zero_pad(pair.full_target_tokens, target_length) for pair in target_batch]))

    if use_cuda:
        input_variable = input_variable.cuda()
        full_input_variable = full_input_variable.cuda()
        target_variable = target_variable.cuda()
        full_target_variable = full_target_variable.cuda()

    encoder_hidden = encoder.initHidden(batch_size)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token] for i in range(batch_size)]))
    if use_cuda: decoder_input = decoder_input.cuda()
    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)

    for token_i in range(target_length):
        decoder_hidden, p_gen, p_vocab, attention_dist = decoder(decoder_input, decoder_hidden, encoder_outputs)

        token_input_dist = Variable(torch.zeros((batch_size, vocab_size+250)))
        padding_matrix = Variable(torch.zeros(batch_size, 250))

        if use_cuda:
            token_input_dist = token_input_dist.cuda()
            padding_matrix = padding_matrix.cuda()

        token_input_dist.scatter_add_(1, full_input_variable, attention_dist)

        p_final = torch.cat((p_vocab * p_gen, padding_matrix), 1) + (1-p_gen) * token_input_dist
        loss += criterion(F.log_softmax(p_final, dim=1), full_target_variable.narrow(1, token_i, 1).squeeze(-1))
        decoder_input = target_variable.narrow(1, token_i, 1) # Teacher forcing


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    print("Batch", batch_nb,"Loss", loss.data[0], 'time', time.time() - last)



for i in range(10000):
    train_batch(i)