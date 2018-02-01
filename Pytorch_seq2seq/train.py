from __future__ import unicode_literals, print_function, division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle

use_cuda = False#torch.cuda.is_available()
multi_gpu = False
from Pytorch_seq2seq.data_loader import Vocab, TextPair, DataSet
data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/'

#if 'dataset' not in globals():
with open(data_path +'DM_25k_summary.pickle', 'rb') as f:
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

    def forward(self, input_token, last_decoder_hidden, encoder_states, full_input_var):
        embedded_input = self.embedding(input_token)
        embedded_input = self.dropout(embedded_input)
        decoder_output, decoder_hidden = self.gru(embedded_input, torch.unsqueeze(last_decoder_hidden, 0))

        att_dist = (decoder_hidden.squeeze(0).unsqueeze(1) * (self.w_h * encoder_states)).sum(-1)

        # att_dist = F.softmax(att_dist, dim=-1)

        temporal_att_dist = None

        context_vector = (torch.unsqueeze(att_dist, 2) * encoder_states).sum(1)
        decoder_context = torch.cat((torch.squeeze(decoder_output, 1), context_vector), -1)
        p_vocab = self.out_vocab(self.out_hidden(decoder_context)) # replace with embedding weight

        p_gen = F.sigmoid(self.gen_layer(torch.cat((decoder_context, torch.squeeze(embedded_input, 1)), 1)))
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

        #print((p_final_2 - p_final).sum(-1))
        return decoder_hidden.squeeze(0), p_final, p_gen, p_vocab, att_dist
        #return decoder_hidden.squeeze(0), None, None, p_vocab, att_dist


    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result


SOS_token = 1
EOS_token = 2
UNK_token = 3

batch_size = 3
embedding_size = 4
hidden_size = 8
input_length = 5
target_length = 2
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


def print_attention_dist(text_pair, vocab, att_dist):
    attentions = []
    for i in range(min(len(att_dist), len(text_pair.full_source_tokens))):
        word_idx = text_pair.full_source_tokens[i]
        if att_dist[i] > 0.05:
            if word_idx < len(vocab.index2word) : attentions.append((vocab.index2word[word_idx], att_dist[i]))
            else:
                for w in text_pair.unknown_tokens.keys():
                    if text_pair.unknown_tokens[w] == word_idx:
                        attentions.append((w, att_dist[i]))
                        break

        '''
        sample = 6
        pred = p_final.max(1)[1][sample].data[0]
        true = full_target_variable.narrow(1, token_i, 1).squeeze(-1)[sample].data[0]
        index2word = dataset.vocab.index2word
        if pred in index2word: pred_w = index2word[pred]
        else: pred_w = 'UNK'
        if true in index2word: true_w = index2word[true]
        else: true_w = 'UNK'
        print(print_attention_dist(samples[sample], dataset.vocab, token_input_dist[0].data))
        print(pred_w, true_w)
        '''

    return attentions



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
    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)

    for token_i in range(target_length):

        decoder_hidden, p_final, p_gen, p_vocab, attention_dist = \
            decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable)
        loss += criterion(F.log_softmax(p_vocab, dim=1), target_variable.narrow(1, token_i, 1).squeeze(-1))
        decoder_input = target_variable.narrow(1, token_i, 1) # Teacher forcing

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


import sys
def progress_bar(fraction):
    sys.stdout.write('\r')
    sys.stdout.write("[%-60s] %d%%" % ('='*int((60*(e+1)/10)), (100*(e+1)/10)))
    sys.stdout.flush()
    sys.stdout.write(", epoch %d" % (e+1))
    sys.stdout.flush()


def translate_word(token, text_pair, vocab):
    print(token)
    if token in vocab.index2word: return vocab.index2word[token]
    if token in text_pair.unknown_tokens.values():
        return [k for k in text_pair.unknown_tokens if text_pair.unknown_tokens[k] == token][0]
    return UNK_token


def predict_and_print(pair):
    print(pair.target_text)
    encoder_hidden = Variable(torch.zeros(2, 1, embedding_size))
    input_variable = Variable(torch.LongTensor([zero_pad(pair.masked_source_tokens, input_length)]))
    full_input_variable = Variable(torch.LongTensor([zero_pad(pair.full_source_tokens, input_length)]))
    if use_cuda:
        input_variable = input_variable.cuda()
        encoder_hidden= encoder_hidden.cuda()
        full_input_variable = full_input_variable.cuda()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if use_cuda: decoder_input = decoder_input.cuda()
    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)

    result = []
    gen_sequence = []
    for token_i in range(target_length):

        decoder_hidden, p_final, p_gen, p_vocab, attention_dist = decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable)
        '''
        p_word, decoded_word_idx = p_final.max(1)
        decoded_word = translate_word(decoded_word_idx.data[0], pair, dataset.vocab)
        p_att, attended_pos = attention_dist.max(1)
        attended_word = translate_word(pair.full_source_tokens[attended_pos.data[0]], pair, dataset.vocab)

        if decoded_word_idx.data[0] < 25000: decoder_input = Variable(torch.LongTensor([[decoded_word_idx.data[0]]]))
        else: decoder_input = Variable(torch.LongTensor([[UNK_token]]))
        if use_cuda: decoder_input = decoder_input.cuda()

        result.append({'p_gen': round(p_gen.data[0][0], 3),'word': decoded_word, 'p_word': round(p_word.data[0], 3),
                      'att_word': attended_word, 'p_att': round(p_att.data[0], 3)})
        '''
        p_vocab_word, vocab_word_idx = p_vocab.max(1)

        gen_sequence.append((translate_word(vocab_word_idx.data[0], pair, dataset.vocab), round(p_vocab_word.data[0], 3)))

    return result, gen_sequence

#predict_and_print(training_pairs[20])


nb_epochs = 20
for e in range(nb_epochs):

    epoch_loss = 0
    for b in range(int(len(training_pairs)/batch_size)):
        batch = training_pairs[b*batch_size:(b+1)*batch_size]
        epoch_loss += train_batch(b, batch)
        if b % 50 == 0:
            print(b*batch_size, '/', len(training_pairs), "loss:", epoch_loss/ (b+1))
            sample = b*batch_size
            test_result, gen_seq = predict_and_print(training_pairs[sample])
            print(gen_seq)













