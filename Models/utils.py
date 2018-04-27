
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


def get_most_attn_word(text_pair, att_dist, vocab):
    max_value, max_token = att_dist.topk(k=3, dim=-1)
    tokens = [text_pair.full_source_tokens[max_token.data[i]] for i in range(3)]
    tuples = [(max_token.data[i], text_pair.get_text([tokens[i]], vocab), round(max_value.data[i], 3)) for i in range(3)]
    return tuples


def get_batch_variables(samples, input_length, target_length, use_cuda, SOS_token):

    input_variable = Variable(torch.LongTensor([zero_pad(pair.masked_source_tokens, input_length) for pair in samples]))
    full_input_variable = Variable(torch.LongTensor([zero_pad(pair.full_source_tokens, input_length) for pair in samples]))
    target_variable = Variable(torch.LongTensor([zero_pad(pair.masked_target_tokens, target_length) for pair in samples]))
    full_target_variable = Variable(torch.LongTensor([zero_pad(pair.full_target_tokens, target_length) for pair in samples]))
    decoder_input = Variable(torch.LongTensor([[SOS_token] for i in range(len(samples))]))
    control_zero = Variable(torch.FloatTensor([[0] for i in range(len(samples))]))

    if use_cuda:
        return input_variable.cuda(), full_input_variable.cuda(), target_variable.cuda(), \
               full_target_variable.cuda(), decoder_input.cuda(), control_zero.cuda()
    else:
        return input_variable, full_input_variable, target_variable, full_target_variable, decoder_input, control_zero


def sort_and_shuffle_data(samples, nb_buckets, batch_size, rnd=True):
    sorted_samples = sorted(samples, key=lambda pair: len(pair.masked_target_tokens))
    bucket_size = int(len(samples) / nb_buckets)
    buckets = [sorted_samples[bucket_size*i:bucket_size*(i+1)] for i in range(nb_buckets)]
    if rnd: random.shuffle(buckets)
    batches = []
    for b in buckets:
        if rnd: random.shuffle(b)
        batches += [b[batch_size*i:batch_size*(i+1)] for i in range(int(len(b) / batch_size))]
    if rnd: random.shuffle(batches)
    return batches






class TrainingLogger():
    def __init__(self, nb_epochs, batch_size, sample_size, val_size):
        self.epoch_nb = 0
        self.log = dict()
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.epoch_size = sample_size
        self.val_size = val_size
        self.epoch_iter = 0
        self.init_epoch(0)

    def add_iteration_v2(self, iter, loss, _time):


        self.log[self.epoch_nb]["loss"] += loss
        self.log[self.epoch_nb]["time"] += _time

        samples_trained = (self.epoch_iter * self.batch_size) % self.epoch_size
        time_per_sample = self.log[self.epoch_nb]["time"] / samples_trained
        remaining_time = (self.epoch_size - samples_trained) * time_per_sample
        self.progress_bar(self.epoch_iter, self.epoch_size / self.batch_size,
                          self.log[self.epoch_nb]["loss"] / self.epoch_iter,
                          remaining_time, self.epoch_nb)



    def add_iteration(self, iter, loss, _time):
        if iter * self.batch_size > (self.epoch_nb * self.epoch_size):
            self.init_epoch(self.epoch_nb + 1)
        self.epoch_iter += 1

        self.log[self.epoch_nb]["loss"] += loss
        self.log[self.epoch_nb]["time"] += _time
        predicted_time = ((self.epoch_size / self.batch_size) / self.epoch_iter) * self.log[self.epoch_nb]["time"]
        remaining_time = predicted_time - self.log[self.epoch_nb]["time"]
        self.progress_bar(self.epoch_iter, self.epoch_size/self.batch_size, self.log[self.epoch_nb]["loss"]/self.epoch_iter, remaining_time,
                          self.epoch_nb)

    def add_val_iteration(self, step, loss, _time):
        self.log[self.epoch_nb]["val_loss"] += loss
        self.log[self.epoch_nb]["val_time"] += _time


    def init_epoch(self, epoch):
        print("Epoch: ", epoch)
        self.epoch_nb = epoch
        self.log[epoch] = {"loss": 0, 'time': 0, "val_loss": 0, "val_time": 0}
        self.epoch_iter = 0


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

        def check_trigram(tri, seq):
            for i in range(3):
                if tri[i] != seq[i]: return False
            return True

        def check_repeats():
            for i in range(max(0, len(self.sequence) - 4), -1, -1):
                if len(self.sequence) > 3 and check_trigram(self.sequence[-3:], self.sequence[i:i + 3]): return True

        if check_repeats(): return 1e7

        for p in [-math.log(log_prob) for log_prob in self.log_probs]:
            score += p
        return score / len(self.sequence)



def translate_word(token, text_pair, vocab):
    if token in vocab.index2word: return vocab.index2word[token]
    if token in text_pair.unknown_tokens.values():
        return [k for k in text_pair.unknown_tokens if text_pair.unknown_tokens[k] == token][0]
    return 3


