from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random
import utils as utils
import shutil
#

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
    def __init__(self, hidden_size, embedding_size, vocab_size,
                 n_layers=1, dropout_p=0.1, encoder_max_length=100, decoder_max_length=100, embedding_weight=None):
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
        self.w_d = nn.Parameter(torch.randn(self.hidden_size))

        # Generator

        self.out_hidden = nn.Linear(self.hidden_size*3, self.embedding_size)
        self.out_vocab = nn.Linear(self.embedding_size, vocab_size)
        self.gen_layer = nn.Linear(self.hidden_size*3, 1)

        # Weight_sharing
        if embedding_weight is not None:
            self.out_vocab.weight = embedding_weight

    def forward(self, input_token, prev_decoder_h_states, last_hidden, encoder_states, full_input_var, previous_att, use_cuda):
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
        '''
        pointer_dist = att_dist * (1-p_gen)
        padding_matrix = Variable(torch.zeros(batch_size, 250)).cuda()
        generator_dist = torch.cat((p_vocab * p_gen, padding_matrix), 1)
        p_final = generator_dist.scatter_add_(1, full_input_var, pointer_dist)
        '''
        token_input_dist = Variable(torch.zeros((full_input_var.size()[0], self.vocab_size+500)))
        padding_matrix_2 = Variable(torch.zeros(full_input_var.size()[0], 500))
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


class PGCModel():
    def __init__(self, config, vocab, model_id, model_path, use_cuda):
        self.use_cuda = use_cuda
        self.config = config
        self.vocab = vocab
        self.model_id = model_id
        self.model_path = model_path

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.input_length = config['input_length']
        self.target_length = config['target_length']

        self.encoder = EncoderRNN(self.vocab.vocab_size, hidden_size=self.embedding_size)
        self.emb_w = self.encoder.embedding.weight # use weight sharing?
        self.decoder = AttnDecoderRNN(self.hidden_size, self.embedding_size, self.vocab.vocab_size, 1,
                                        dropout_p=0.1, embedding_weight=None)
        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.criterion = None
        self.logger = None
        print("Model compiled")


    def train(self, data, val_data, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion, use_cuda, _print):

        if self.logger is None:
            self.encoder_optimizer = optimizer(self.encoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.decoder_optimizer = optimizer(self.decoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.criterion = nn.NLLLoss()
            self.logger = utils.TrainingLogger(nb_epochs, batch_size, len(data))
            print("Optimizers compiled")

        for epoch in range(len(self.logger.log), nb_epochs):
            random.shuffle(data)
            self.logger.init_epoch(epoch)
            for b in range(int(len(data)/batch_size)):
                loss, _time = self.train_batch(samples=data[b*batch_size:(b+1)*batch_size], use_cuda=self.use_cuda)
                self.logger.add_iteration(b+1, loss, _time)
                if b % 200 == 0 and _print:
                    preds = self.predict([data[b*batch_size]], self.target_length, False, self.use_cuda)
                    print('\n', [(t[0]['word'], t[0]['p_gen']) for t in preds])
                    print(" ".join([t[0]['word'] for t in preds]))

            for b in range(int(len(data)/batch_size)):
                loss, _time = self.train_batch(val_data[b*batch_size:(b+1)*batch_size], self.use_cuda, backprop=False)
                self.logger.add_val_iteration(b+1, loss, _time)

            if epoch == 0 or self.logger.log[epoch]["val_loss"] < self.logger.log[epoch-1]["val_loss"]:
                self.save_model(self.model_path, self.model_id, epoch=epoch, loss=self.logger.log[epoch]["val_loss"])


    def train_batch(self, samples, use_cuda, tf_ratio=0.5, backprop=True):
        start = time.time()
        if len(samples) == 0: return 0, 0
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.input_length, self.target_length, use_cuda,
                                      self.vocab.word2index['SOS'])

        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0
        if not backprop: print(samples, input_variable)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_hidden_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        for token_i in range(self.target_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs, full_input_variable, previous_att, use_cuda)
            loss += self.criterion(torch.log(p_final.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1).squeeze(-1))

            if random.uniform(0, 1) < tf_ratio: decoder_input = target_variable.narrow(1, token_i, 1)
            else:
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if max_tokens.data[i] >= self.vocab.vocab_size: max_tokens.data[i] = self.vocab.word2index['UNK']
                decoder_input = max_tokens.unsqueeze(1)

        if backprop:
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        return loss.data[0] / self.target_length, time.time() - start



    def predict(self, samples, target_length, beam_size, use_cuda): # this only works with one sample at a time
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.input_length, target_length, use_cuda,
                                      self.vocab.word2index['SOS'])
        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_h_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        if not beam_size:
            result = []
            for token_i in range(target_length):

                p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                    self.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs, full_input_variable, previous_att, use_cuda)

                if not beam_size:
                    p_vocab_word, vocab_word_idx = p_final.max(1)
                    result.append([{'token_idx': vocab_word_idx.data[i],
                                    'word': utils.translate_word(vocab_word_idx.data[i], samples[i], self.vocab),
                                    'p_gen': round(p_gen.data[i][0], 3)}
                                        for i in range(len(samples))])
                    _, max_tokens = p_final.max(1)
                    for i in range(max_tokens.size()[0]):
                        if max_tokens.data[i] >= self.vocab.vocab_size: max_tokens.data[i] = self.vocab.word2index['UNK']
                    decoder_input = max_tokens.unsqueeze(1)

                else:
                    pass
                    # conduct beam search
            return result

        else:
            search_complete = False
            top_beams = [Beam(decoder_input, decoder_h_states, decoder_hidden, previous_att, [], [])]

            while not search_complete:
                new_beams = []
                for beam in top_beams:
                    if beam.complete: new_beams.append(beam)
                    else:
                        p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                            self.decoder(beam.decoder_input, beam.decoder_h_states, beam.decoder_hidden,
                                         encoder_outputs, full_input_variable, beam.previous_att, use_cuda)
                        for k in range(beam_size):
                            p_vocab_word, vocab_word_idx = p_final.max(1)
                            _, max_tokens = p_final.max(1)
                            if max_tokens.data[0] >= self.vocab.vocab_size: max_tokens.data[0] = self.vocab.word2index['UNK']
                            new_beams.append(Beam(max_tokens.unsqueeze(1),
                                                  decoder_h_states, decoder_hidden, previous_att,
                                                  beam.log_probs+[p_vocab_word.data[0]],
                                                  beam.sequence + [vocab_word_idx.data[0]]))
                            p_final[0, vocab_word_idx.data[0]] = 0

                            if len(new_beams[-1].sequence) == target_length or vocab_word_idx.data[0] == self.vocab.word2index['EOS']:
                                new_beams[-1].complete = True

                all_beams = sorted([(b, b.compute_score()) for b in new_beams], key=lambda tup: tup[1])
                if len(all_beams) > beam_size: all_beams = all_beams[:beam_size]
                top_beams = [beam[0] for beam in all_beams]

                if len([True for b in top_beams if b.complete]) == beam_size: search_complete = True

            return [[" ".join([utils.translate_word(t, samples[0], self.vocab) for t in b.sequence]),
                     b.compute_score()]
                    for b in top_beams]





    def save_model(self, path, id, epoch, loss):
        data = {
            'epoch': epoch + 1,
            'best_prec1': loss,
            'vocab': self.vocab,
            'config': self.config,
            'logger': self.logger,
            'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict(),
            'encoder_optm': self.encoder_optimizer.state_dict(),'decoder_optm': self.decoder_optimizer.state_dict()
        }
        filename= path + "checkpoint_" + id + "_ep@.pickle"
        torch.save(data, filename)

    def load_model(self, file_path, file_name):
        data = torch.load(file_path + file_name)
        self.encoder.load_state_dict(data['encoder'])
        self.decoder.load_state_dict(data['decoder'])
        self.vocab = data['vocab']

import math
class Beam():
    def __init__(self, decoder_input, decoder_h_states, decoder_hidden, previous_att, log_probs, sequence):
        self.decoder_input = decoder_input
        self.decoder_hidden = decoder_hidden
        self.decoder_h_states = decoder_h_states
        self.previous_att = previous_att

        self.log_probs = log_probs
        self.sequence = sequence
        self.complete = False

    def compute_score(self):
        score = 1
        for p in [-math.log(log_prob) for log_prob in self.log_probs]:
            score += p
        return score











