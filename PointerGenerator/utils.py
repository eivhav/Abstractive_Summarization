
from __future__ import unicode_literals, print_function, division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import math
import random


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


def get_batch_variables(samples, input_length, target_length, use_cuda, SOS_token):

    input_variable = Variable(torch.LongTensor([zero_pad(pair.masked_source_tokens, input_length) for pair in samples]))
    full_input_variable = Variable(torch.LongTensor([zero_pad(pair.full_source_tokens, input_length) for pair in samples]))
    target_variable = Variable(torch.LongTensor([zero_pad(pair.masked_target_tokens, target_length) for pair in samples]))
    full_target_variable = Variable(torch.LongTensor([zero_pad(pair.full_target_tokens, target_length) for pair in samples]))
    decoder_input = Variable(torch.LongTensor([[SOS_token] for i in range(len(samples))]))

    if use_cuda:
        return input_variable.cuda(), full_input_variable.cuda(), target_variable.cuda(), \
               full_target_variable.cuda(), decoder_input.cuda()
    else:
        return input_variable, full_input_variable, target_variable, full_target_variable, decoder_input


def sort_and_shuffle_data(samples, nb_buckets, batch_size, rnd=True):
    sorted_samples = sorted(samples, key=lambda pair: len(pair.masked_target_tokens))
    bucket_size = int(len(samples) / nb_buckets)
    buckets = [sorted_samples[bucket_size*i:bucket_size*(i+1)] for i in range(nb_buckets)]
    if rnd: random.shuffle(buckets)
    batches = []
    for b in buckets:
        if rnd: random.shuffle(b)
        batches += [b[batch_size*i:batch_size*(i+1)] for i in range(int(len(b) / batch_size))]
    return batches






class TrainingLogger():
    def __init__(self, nb_epochs, batch_size, sample_size, val_size):
        self.epoch_nb = 0
        self.log = dict()
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.epoch_size = sample_size
        self.val_size = val_size


    def add_iteration(self, step, loss, _time):
        self.log[self.epoch_nb]["loss"] += loss
        self.log[self.epoch_nb]["time"] += _time
        predicted_time = ((self.epoch_size / self.batch_size) / step) * self.log[self.epoch_nb]["time"]
        remaining_time = predicted_time - self.log[self.epoch_nb]["time"]
        self.progress_bar(step, self.epoch_size/self.batch_size, self.log[self.epoch_nb]["loss"]/step, remaining_time,
                          self.epoch_nb)

    def add_val_iteration(self, step, loss, _time):
        self.log[self.epoch_nb]["val_loss"] += loss
        self.log[self.epoch_nb]["val_time"] += _time


    def init_epoch(self, epoch):
        if epoch > 0: print("Epoch complete. Validation loss: ", self.batch_size* self.log[epoch -1]["val_loss"] / self.val_size)
        print("Epoch: ", epoch)
        self.epoch_nb = epoch
        self.log[epoch] = {"loss": 0, 'time': 0, "val_loss": 0, "val_time": 0}

    def progress_bar(self, step_nb, nb_steps, loss, time_left, e):
        sys.stdout.write('\r')
        sys.stdout.write("%d/%d" % (step_nb*self.batch_size, self.epoch_size))
        sys.stdout.flush()
        sys.stdout.write("[%-60s] %d%%" % ('='*int(60*(step_nb/nb_steps)), (100*step_nb/nb_steps)))
        sys.stdout.flush()
        sys.stdout.write(", time_left %d" % time_left)
        sys.stdout.flush()
        sys.stdout.write(", loss: " + str(loss))
        sys.stdout.flush()



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






def translate_word(token, text_pair, vocab):
    if token in vocab.index2word: return vocab.index2word[token]
    if token in text_pair.unknown_tokens.values():
        return [k for k in text_pair.unknown_tokens if text_pair.unknown_tokens[k] == token][0]
    return 3


def predict_and_print(pair, encoder, decoder, input_length, target_length, SOS_token, vocab, use_cuda, UNK_token):
    print(pair.get_text(pair.full_target_tokens, vocab))

    input_variable, full_input_variable, _, _, decoder_input = \
        get_batch_variables([pair], input_length, target_length, use_cuda, SOS_token)

    encoder_hidden = encoder.init_hidden(1, use_cuda)

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
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

        gen_sequence.append((translate_word(vocab_word_idx.data[0], pair, vocab), round(p_vocab_word.data[0], 3)))

    return result, gen_sequence

#predict_and_print(training_pairs[20])