from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random
import PGC.utils as utils


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        return self.gru(embedded, hidden)

    def init_hidden(self, batch_size, use_cuda):
        if use_cuda: return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()
        else: return Variable(torch.zeros(2, batch_size, self.hidden_size))


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, n_layers=1, dropout_p=0.1, embedding_weight=None):
        super(AttnDecoderRNN, self).__init__()
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

    def forward(self, input_token, last_decoder_hidden, encoder_states, full_input_var, use_cuda):
        embedded_input = self.embedding(input_token)
        embedded_input = self.dropout(embedded_input)
        decoder_output, decoder_hidden = self.gru(embedded_input, torch.unsqueeze(last_decoder_hidden, 0))

        att_dist = F.tanh((self.w_h * encoder_states) + (self.w_s * decoder_output) + self.att_bias)
        att_dist = (self.attn_weight_v * att_dist).sum(-1)
        att_dist = F.softmax(att_dist, dim=-1)

        context_vector = (torch.unsqueeze(att_dist, 2) * encoder_states).sum(1)
        decoder_context = torch.cat((torch.squeeze(decoder_output, 1), context_vector), -1)
        p_vocab = F.softmax(self.out_vocab(self.out_hidden(decoder_context)), dim=-1) # replace with embedding weight

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

    def init_hidden(self, use_cuda):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda: return result.cuda()
        else: return result



class PGCModel():
    def __init__(self, config, vocab, use_cuda):
        self.use_cuda = True
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(vocab.index2word)

        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        self.embedding_size = 128
        self.hidden_size = 256
        self.input_length = 300
        self.target_length = 40

        self.encoder = EncoderRNN(self.vocab_size, hidden_size=self.embedding_size)
        self.emb_w = self.encoder.embedding.weight # use weight sharing?
        self.decoder = AttnDecoderRNN(self.hidden_size, self.embedding_size, self.vocab_size, 1,
                                        dropout_p=0.1, embedding_weight=None)
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.criterion = None

    def train(self, data, val_data, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion, use_cuda):

        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr= lr, weight_decay=0.0000001)
        self.decoder_optimizer = optimizer(self.decoder.parameters(), lr= lr, weight_decay=0.0000001)
        self.criterion = nn.NLLLoss()
        logger = utils.TrainingLogger()

        for epoch in range(nb_epochs):
            random.shuffle(data)
            logger.init_epoch(epoch, nb_epochs)
            for b in range(int(len(data)/batch_size)):
                loss, _time = self.train_batch(samples=data[b*batch_size:(b+1)*batch_size], use_cuda=self.use_cuda)
                logger.add_iteration(b, loss, _time)


    def train_batch(self, samples, use_cuda):
        start = time.time()
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.input_length, self.target_length, use_cuda, self.SOS_token)

        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0

        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)

        for token_i in range(self.target_length):
            decoder_hidden, p_final, p_gen, p_vocab, attention_dist = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable, use_cuda)
            loss += self.criterion(torch.log(p_final.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1).squeeze(-1))
            decoder_input = target_variable.narrow(1, token_i, 1) # Teacher forcing

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / self.target_length, time.time() - start


    def predict(self, samples, target_length, beam_size, use_cuda):
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.input_length, target_length, use_cuda, self.SOS_token)
        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)

        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)

        result = []
        for token_i in range(target_length):

            decoder_hidden, p_final, p_gen, p_vocab, attention_dist = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable, use_cuda)

            if not beam_size:
                p_vocab_word, vocab_word_idx = p_final.max(1)
                result.append([{'token_idx': vocab_word_idx.data[i],
                                'word': utils.translate_word(vocab_word_idx.data[i], samples[i], self.vocab),
                                'p_gen': round(p_gen.data[i][0], 3)}
                                    for i in range(len(samples))])
                decoder_input = torch.unsqueeze(vocab_word_idx, 1)

                #decoder_input = target_variable.narrow(1, token_i, 1) # Teacher forcing
            else:
                pass
                # conduct beam search
        return result













