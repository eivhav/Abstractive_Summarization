from __future__ import unicode_literals, print_function, division

import random
import time

#import utils as utils
#from utils import *
#from pointer_gen import *
import PointerGenerator.utils as utils
from PointerGenerator.utils import *
from PointerGenerator.pointer_gen import *


class PGModel():
    def __init__(self, config, vocab, use_cuda):
        self.use_cuda = use_cuda
        self.config = config
        self.vocab = vocab

        self.encoder = EncoderRNN(self.vocab.vocab_size, hidden_size=config['embedding_size'])
        self.emb_w = self.encoder.embedding.weight # use weight sharing?

        if config['model_type'] == 'TemporalAttn':
            self.decoder = TemporalAttnDecoderRNN(config['hidden_size'], config['embedding_size'],
                                                   self.vocab.vocab_size, 1, dropout_p=0.1, embedding_weight=None)
        else:
            self.decoder = CoverageAttnDecoderRNN(config['hidden_size'], config['embedding_size'],
                                                  self.vocab.vocab_size, 1, dropout_p=0.1,
                                                  input_lenght=config['input_length'], embedding_weight=None)
        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.criterion = None
        self.logger = None
        print("Model compiled")


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


    def train(self, data, val_data, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion, use_cuda, print_evry):

        if self.logger is None:
            self.encoder_optimizer = optimizer(self.encoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.decoder_optimizer = optimizer(self.decoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.criterion = nn.NLLLoss()
            self.logger = TrainingLogger(nb_epochs, batch_size, len(data), len(val_data))
            print("Optimizers compiled")

        for epoch in range(len(self.logger.log), nb_epochs):
            random.shuffle(data)
            self.logger.init_epoch(epoch)
            batches = utils.sort_and_shuffle_data(data, nb_buckets=100, batch_size=batch_size)
            for b in range(len(batches)):
                loss, _time = self.train_batch(samples=batches[b], use_cuda=self.use_cuda)
                self.logger.add_iteration(b+1, loss, _time)
                if b % print_evry == 0:
                    preds = self.predict([data[b*batch_size]], self.config['target_length'], False, self.use_cuda)
                    print('\n', " ".join([t[0]['word'] for t in preds]))

            for b in range(int(len(val_data)/batch_size)):
                loss, _time = self.train_batch(val_data[b*batch_size:(b+1)*batch_size], self.use_cuda, backprop=False)
                self.logger.add_val_iteration(b+1, loss, _time)

            if epoch == 0 or self.logger.log[epoch]["val_loss"] < self.logger.log[epoch-1]["val_loss"]:
                self.save_model(self.config['model_path'], self.config['model_id'],
                                epoch=epoch, loss=self.logger.log[epoch]["val_loss"])


    def train_batch(self, samples, use_cuda, tf_ratio=0.5, backprop=True, coverage_lambda=-1):
        start = time.time()
        if len(samples) == 0: return 0, 0

        target_length = max([len(pair.masked_target_tokens) for pair in samples])
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.config['input_length'], target_length, use_cuda,
                                      self.vocab.word2index['SOS'])

        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0

        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_hidden_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        for token_i in range(min(target_length, self.config['target_length'])):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs, full_input_variable, previous_att, use_cuda)

            if coverage_lambda < 0 or token_i == 0:
                loss += self.criterion(torch.log(p_final.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1)
                                       .squeeze(-1))
            else:
                coverage = previous_att.narrow(1, 0, previous_att.size()[1]-1).sum(dim=1)
                coverage_min, _ = torch.cat((att_dist.unsqueeze(1), coverage.unsqueeze(1)), dim=1).min(dim=1)
                coverage_loss = coverage_min.sum(-1)
                loss += self.criterion(torch.log(p_final.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1).squeeze(-1))\
                        + (coverage_lambda * coverage_loss) # this needs to be fixed

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

        return loss.data[0] / target_length, time.time() - start



    def predict(self, samples, target_length, beam_size, use_cuda): # this only works with one sample at a time
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
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
                    self.decoder(decoder_input, decoder_h_states, decoder_hidden,
                                 encoder_outputs, full_input_variable, previous_att, use_cuda)

                p_vocab_word, vocab_word_idx = p_final.max(1)
                result.append([{'token_idx': vocab_word_idx.data[i],
                                'word': utils.translate_word(vocab_word_idx.data[i], samples[i], self.vocab),
                                'p_gen': round(p_gen.data[i][0], 3)}
                                    for i in range(len(samples))])
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if max_tokens.data[i] >= self.vocab.vocab_size: max_tokens.data[i] = self.vocab.word2index['UNK']
                decoder_input = max_tokens.unsqueeze(1)

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



