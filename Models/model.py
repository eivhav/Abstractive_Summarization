from __future__ import unicode_literals, print_function, division

import random
import time

import Models.utils as utils
from Models.utils import *
from Models.pointer_gen import *


class PGModel():
    def __init__(self, config, vocab, use_cuda, model_file=None):
        self.use_cuda = use_cuda
        self.config = config
        self.vocab = vocab
        self.encoder = None
        self.decoder = None

        if model_file is None:
            self.init_model()
            self.encoder_optimizer = None
            self.decoder_optimizer = None
            self.logger = None
        else:
            data = torch.load(model_file)
            if config is None: self.config = data['config']
            self.init_model()
            self.encoder.load_state_dict(data['encoder'])
            self.decoder.load_state_dict(data['decoder'])
            self.vocab = data['vocab']
            self.encoder_optimizer = data['encoder_optm']
            self.decoder_optimizer = data['decoder_optm']
            self.logger = data['logger']

        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

        self.criterion = None
        print("Model compiled")

    def init_model(self):
        self.encoder = EncoderRNN(self.vocab.vocab_size, self.config['embedding_size'],
                                    hidden_size=self.config['hidden_size'])

        if 'out_hidden_size' not in self.config: self.config['out_hidden_size'] = 128
        self.decoder = ComboAttnDecoderRNN(self.config['hidden_size'], self.config['embedding_size'],
                                           self.vocab.vocab_size, 1,
                                           dropout_p=0.0,
                                           input_lenght=self.config['input_length'],
                                           embedding_weight=None,
                                           temporal_att= self.config['temporal_att'],
                                           bilinear_attn= self.config['bilinear_attn'],
                                           decoder_att=self.config['decoder_att'],
                                           input_in_pgen=self.config['input_in_pgen'],
                                           out_hidden_size=self.config['out_hidden_size'])


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
        filename= path + "checkpoint_" + id + "_ep@"+str(epoch)+"_loss@"+str(round(loss, 3))+".pickle"
        torch.save(data, filename)

    def load_model(self, file_path, file_name):
        data = torch.load(file_path + file_name)
        self.encoder.load_state_dict(data['encoder'])
        self.decoder.load_state_dict(data['decoder'])
        self.vocab = data['vocab']


    def predict(self, samples, target_length, beam_size, use_cuda):
        nb_unks = max([len(s.unknown_tokens) for s in samples])
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input, control_zero = \
            utils.get_batch_variables(samples, self.config['input_length'], target_length, use_cuda,
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
                    self.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs, full_input_variable,
                                 previous_att, control_zero, nb_unks, use_cuda)

                p_vocab_word, vocab_word_idx = p_final.max(1)
                result.append([{'token_idx': vocab_word_idx.data[i],
                                'word': utils.translate_word(vocab_word_idx.data[i], samples[i], self.vocab),
                                'p_gen': round(p_gen.data[i][0], 3)}
                                #'attn': utils.get_most_attn_word(samples[i], att_dist[i], self.vocab)}
                                    for i in range(len(samples))])
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if max_tokens.data[i] >= self.vocab.vocab_size: max_tokens.data[i] = self.vocab.word2index['UNK']
                decoder_input = max_tokens.unsqueeze(1)

            return result

        else: # beam only works one sample at a time
            search_complete = False
            top_beams = [Beam(decoder_input, decoder_h_states, decoder_hidden, previous_att, [], [])]

            def predict_for_beams(beams, encoder_outputs, full_input_variable, control_var):

                results = []

                encoder_outputs = torch.stack([encoder_outputs[i] for beam in beams for i in range(len(samples))], 0)
                full_input_variable = torch.stack([full_input_variable[i] for beam in beams for i in range(len(samples))], 0)
                decoder_input = torch.stack([beam.decoder_input[i] for beam in beams for i in range(len(samples))], 0)
                decoder_h_states = torch.stack([beam.decoder_h_states[i] for beam in beams for i in range(len(samples))], 0)
                decoder_hidden = torch.stack([beam.decoder_hidden[i] for beam in beams for i in range(len(samples))], 0)
                control_var = torch.stack([control_var[i] for beam in beams for i in range(len(samples))], 0)

                if beams[0].previous_att is not None:
                    previous_att = torch.stack([beam.previous_att[i] for beam in beams for i in range(len(samples))], 0)
                else: previous_att = None

                p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                    self.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs, full_input_variable,
                                 previous_att, control_var, nb_unks, use_cuda)

                for b in range(len(beams)):
                    results.append([beams[b]] + [tensor.narrow(0, b*len(samples), len(samples)) for tensor in
                                [p_final, decoder_h_states, decoder_hidden, previous_att]])

                return results

            while not search_complete:
                new_beams = []
                beams_to_predict = []

                for beam in top_beams:
                    if beam.complete:
                        new_beams.append(beam)
                    else:
                        beams_to_predict.append(beam)

                predictions = predict_for_beams(beams_to_predict, encoder_outputs, full_input_variable, control_zero)
                for b in predictions:
                    beam = b[0]
                    p_final, decoder_h_states, decoder_hidden, previous_att = b[1], b[2], b[3], b[4]

                    p_top_words, top_indexes = p_final.topk(beam_size)

                    for k in range(beam_size):
                        non_masked_word = top_indexes.data[0][k]
                        if top_indexes.data[0][k] >= self.vocab.vocab_size:
                            top_indexes.data[0][k] = self.vocab.word2index['UNK']

                        new_beams.append(Beam(top_indexes.narrow(1, k, 1),
                                              decoder_h_states, decoder_hidden, previous_att,
                                              beam.log_probs + [p_top_words.data[0][k]],
                                              beam.sequence + [non_masked_word]))

                        if len(new_beams[-1].sequence) == target_length or top_indexes.data[0][k] == \
                                self.vocab.word2index['EOS']:
                            new_beams[-1].complete = True

                all_beams = sorted([(b, b.compute_score()) for b in new_beams], key=lambda tup: tup[1])
                if len(all_beams) > beam_size: all_beams = all_beams[:beam_size]
                top_beams = [beam[0] for beam in all_beams]

                if len([True for b in top_beams if b.complete]) == beam_size: search_complete = True

            return [[" ".join([utils.translate_word(t, samples[0], self.vocab) for t in b.sequence]),
                     b.compute_score()]
                    for b in top_beams]



    def predict_v2(self, samples, target_length, beam_size, use_cuda):
        nb_unks = max([len(s.unknown_tokens) for s in samples])
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input, control_zero = \
            utils.get_batch_variables(samples, self.config['input_length'], target_length, use_cuda,
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
                    self.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs, full_input_variable,
                                 previous_att, control_zero, nb_unks, use_cuda)

                p_vocab_word, vocab_word_idx = p_final.max(1)
                result.append([{'token_idx': vocab_word_idx.data[i],
                                'word': utils.translate_word(vocab_word_idx.data[i], samples[i], self.vocab),
                                'p_gen': round(p_gen.data[i][0], 3)}
                                #'attn': utils.get_most_attn_word(samples[i], att_dist[i], self.vocab)}
                                    for i in range(len(samples))])
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if max_tokens.data[i] >= self.vocab.vocab_size: max_tokens.data[i] = self.vocab.word2index['UNK']
                decoder_input = max_tokens.unsqueeze(1)

            return result

        else: # beam only works one sample at a time

            search_complete = [False for i in range(len(samples))]
            top_beams = [[Beam(decoder_input.narrow(0, i, 1), decoder_h_states.narrow(0, i, 1),
                               decoder_hidden.narrow(0, i, 1), previous_att, [], [], [])] for i in range(len(samples))]

            first = True
            while False in search_complete:
                beams_to_predict = []
                new_beams = []
                for sample in range(len(samples)):
                    new_beams.append([])
                    beams_to_predict.append([])

                    for beam in top_beams[sample]:
                        if beam.complete: new_beams[-1].append(beam)
                        else: beams_to_predict[-1].append(beam)

                predictions = self.predict_for_beams(beams_to_predict, encoder_outputs, full_input_variable,
                                                     control_zero, nb_unks, use_cuda, first)
                first = False

                for sample in range(len(predictions)):
                    for b in predictions[sample]:
                        beam = b[0]

                        p_final, decoder_h_states, decoder_hidden, previous_att, p_gens = b[1], b[2], b[3], b[4], b[5]
                        p_top_words, top_indexes = p_final.topk(beam_size)

                        for k in range(beam_size):
                            non_masked_word = top_indexes.data[0][k]

                            if top_indexes.data[0][k] >= self.vocab.vocab_size:
                                top_indexes.data[0][k] = self.vocab.word2index['UNK']

                            new_beams[sample].append(Beam(top_indexes.narrow(1, k, 1),
                                                  decoder_h_states, decoder_hidden, previous_att,
                                                  beam.log_probs + [p_top_words.data[0][k]],
                                                  beam.sequence + [non_masked_word],
                                                  beam.p_gens + [p_gens.data[0][0]]))

                            if len(new_beams[sample][-1].sequence) == target_length or top_indexes.data[0][k] == \
                                    self.vocab.word2index['EOS']:
                                new_beams[sample][-1].complete = True

                    all_beams = sorted([(b, b.compute_score()) for b in new_beams[sample]], key=lambda tup: tup[1])
                    if len(all_beams) > beam_size:
                        all_beams = all_beams[:beam_size]
                    top_beams[sample] = [beam[0] for beam in all_beams]

                    if len([True for b in top_beams[sample] if b.complete]) == beam_size: search_complete[sample] = True

            return [[[" ".join([str(utils.translate_word(t, samples[sample], self.vocab)) for t in b.sequence]),
                     b.compute_score(), b.avg_p_gens(), b.p_gens, b.previous_att]
                     for b in top_beams[sample]] for sample in range(len(samples))]



    def predict_for_beams(self, batch_beams, encoder_outputs, full_input_variable, control_var, nb_unks, use_cuda, first):

        results = []
        for s in range(len(batch_beams)): results.append([])

        prediction_data = [[], [], [], [], [], [], []]
        for s in range(len(batch_beams)):
            prediction_data[0] += [encoder_outputs[s] for beam in batch_beams[s]]
            prediction_data[1] += [full_input_variable[s] for beam in batch_beams[s]]
            prediction_data[2] += [beam.decoder_input for beam in batch_beams[s]]
            prediction_data[3] += [beam.decoder_h_states for beam in batch_beams[s]]
            prediction_data[4] += [beam.decoder_hidden for beam in batch_beams[s]]
            prediction_data[5] += [control_var[s] for beam in batch_beams[s]]
            if not first:
                prediction_data[6] += [beam.previous_att for beam in batch_beams[s]]

        encoder_outputs = torch.stack(prediction_data[0], 0)
        full_input_variable = torch.stack(prediction_data[1], 0)
        decoder_input = torch.cat(prediction_data[2], 0)
        decoder_h_states = torch.cat(prediction_data[3], 0)
        decoder_hidden = torch.cat(prediction_data[4], 0)
        control_var = torch.stack(prediction_data[5], 0)

        if not first: previous_att = torch.cat(prediction_data[6], 0)
        else: previous_att = None


        p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
            self.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs, full_input_variable,
                         previous_att, control_var, nb_unks, use_cuda)

        idx = 0
        for s in range(len(batch_beams)):
            for beam in batch_beams[s]:
                results[s].append([beam] + [tensor.narrow(0, idx, 1) for tensor in
                                            [p_final, decoder_h_states, decoder_hidden, previous_att, p_gen]])
                idx += 1

        return results
