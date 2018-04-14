from __future__ import unicode_literals, print_function, division

import random
import time

#import utils as utils
#from utils import *
#from pointer_gen import *
import Models.utils as utils
from Models.utils import *
from Models.pointer_gen import *
from tensorboardX import SummaryWriter
from sumeval.metrics.rouge import RougeCalculator

class MLE_Trainer(): 
    def __init__(self, model):
        self.model = model
        self.max_pool = nn.MaxPool2d((3, 1), stride=(1, 1))
        self.writer = SummaryWriter()


    def train(self, data, val_data, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion,
              use_cuda, print_evry, novelty_loss=-1):

        if self.model.logger is None:
            self.model.encoder_optimizer = optimizer(self.model.encoder.parameters(), lr= lr)
            self.model.decoder_optimizer = optimizer(self.model.decoder.parameters(), lr= lr)
            self.model.logger = TrainingLogger(nb_epochs, batch_size, len(data), len(val_data))
            print("Optimizers compiled")

        self.model.criterion = nn.NLLLoss()

        for epoch in range(len(self.model.logger.log), nb_epochs):
            self.model.logger.init_epoch(epoch)
            batches = utils.sort_and_shuffle_data(data, nb_buckets=100, batch_size=batch_size, rnd=True)
            for b in range(len(batches)):
                loss, _time = self.train_batch(samples=batches[b], use_cuda=self.model.use_cuda, novelty_lambda=novelty_loss)
                self.writer.add_scalar('Training Loss', loss, (epoch*len(batches)) + b)
                self.model.logger.add_iteration(b+1, loss, _time)
                if b % print_evry == 0:
                    self.validate_model(data, val_data, batch_size, use_cuda, epoch, batches, b, novelty_loss)

            for b in range(int(len(val_data)/batch_size)):
                try:
                    loss, _time = self.train_batch(val_data[b*batch_size:(b+1)*batch_size], self.model.use_cuda,
                                                   backprop=False, novelty_lambda=novelty_loss)
                    self.model.logger.add_val_iteration(b+1, loss, _time)
                except:
                    print("\n", "Error during validation!")

            if epoch == 0 or self.model.logger.log[epoch]["val_loss"] < self.model.logger.log[epoch-1]["val_loss"]:
                self.model.save_model(self.model.config['model_path'], self.model.config['model_id'],
                                epoch=epoch, loss=self.model.logger.log[epoch]["val_loss"])


    def train_batch(self, samples, use_cuda, tf_ratio=0.5, backprop=True, novelty_lambda=-1):
        start = time.time()
        if len(samples) == 0: return 0, 0

        target_length = min(self.model.config['target_length'], max([len(pair.masked_target_tokens) for pair in samples]))
        nb_unks = max([len(s.unknown_tokens) for s in samples])

        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input, control_var_zero = \
            utils.get_batch_variables(samples, self.model.config['input_length'], target_length, use_cuda,
                                      self.model.vocab.word2index['SOS'])

        encoder_hidden = self.model.encoder.init_hidden(len(samples), use_cuda)
        self.model.encoder_optimizer.zero_grad()
        self.model.decoder_optimizer.zero_grad()
        loss = 0

        encoder_outputs, encoder_hidden = self.model.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_h_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None
        p_gens = None

        for token_i in range(target_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.model.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs,
                             full_input_variable, previous_att, control_var_zero, nb_unks, use_cuda)

            loss += self.model.criterion(torch.log(p_final.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1)
                                       .squeeze(-1))

            if random.uniform(0, 1) < tf_ratio: decoder_input = target_variable.narrow(1, token_i, 1)
            else:
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if max_tokens.data[i] >= self.model.vocab.vocab_size:
                        max_tokens.data[i] = self.model.vocab.word2index['UNK']
                decoder_input = max_tokens.unsqueeze(1)

            if novelty_lambda > 0:
                if token_i == 0: p_gens = p_gen.unsqueeze(-1)
                else: p_gens = torch.cat((p_gens, p_gen.unsqueeze(-1)), 1)

        if novelty_lambda > 0:
            scaled_att = p_gens * previous_att

            _size = scaled_att.size()[2] - 3

            first_tri_gram = [scaled_att.narrow(1, i, 1).narrow(2, i, _size) for i in range(3)]
            novelty_att = torch.cat((first_tri_gram[0], first_tri_gram[1], first_tri_gram[2]), 1).unsqueeze(1)
            for t in range(1, target_length-2):
                tri_gram = [scaled_att.narrow(1, t+i, 1).narrow(2, i, _size) for i in range(3)]
                tri_sequence = torch.cat((tri_gram[0], tri_gram[1], tri_gram[2]), 1).unsqueeze(1)
                novelty_att = torch.cat((novelty_att, tri_sequence), 1)

            min_values = self.max_pool(-novelty_att).squeeze(2)
            novelty_loss = torch.abs(min_values.sum(-1).sum(-1).sum(-1))

            '''
            print('\n', (1 - novelty_lambda) * loss.data[0]/target_length,
                  -novelty_lambda * novelty_loss.data[0]/target_length)
            '''
            loss += novelty_lambda * novelty_loss

        if backprop:
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm(self.model.decoder.parameters(), 2)
            self.model.encoder_optimizer.step()
            self.model.decoder_optimizer.step()

        return loss.data[0] / target_length, time.time() - start



    def validate_model(self, data, val_data, batch_size, use_cuda, epoch, batches, b, novelty_loss):

        val_loss = 0
        val_batches = [val_data[b * batch_size:(b + 1) * batch_size] for b in range(50)]
        for batch in val_batches:
            loss, _time = self.train_batch(batch, self.model.use_cuda, backprop=False, novelty_lambda=novelty_loss)
            val_loss += loss
        self.writer.add_scalar('Validation Loss', val_loss / len(val_batches), (epoch * len(batches)) + b)

        preds = self.model.predict([data[b * batch_size]], self.model.config['target_length'], False, self.model.use_cuda)
        self.writer.add_text('Greedy Prediction', " ".join([t[0]['word'] for t in preds]), (epoch * len(batches)) + b)

        preds_beam = self.model.predict([data[b * batch_size]], self.model.config['target_length'], 5, self.model.use_cuda)
        self.writer.add_text('Beam Prediction', preds_beam[0][0], (epoch * len(batches)) + b)

        r1, r2, r_l = self.score_model(val_data[0:1500], batch_size, use_cuda)
        self.writer.add_scalar('Rouge-1', 100 * r1, (epoch * len(batches)) + b)
        self.writer.add_scalar('Rouge-2', 100 * r2, (epoch * len(batches)) + b)
        self.writer.add_scalar('Rouge-L', 100 * r_l, (epoch * len(batches)) + b)

    def score_model(self, data, batch_size, use_cuda):

        results = dict()
        for i in range(int(len(data) / batch_size)):
            preds = self.model.predict(data[i * batch_size:(i + 1) * batch_size], 75, False, use_cuda)
            for p in range(batch_size):
                pair = data[(i * batch_size) + p]
                ref = pair.get_text(pair.full_target_tokens, self.model.vocab).replace(" EOS", "")
                seq = [t[p] for t in preds]
                arg_max = " ".join([s['word'] for s in seq]).split("EOS")[0]
                #full_text = pair.get_text(pair.full_source_tokens, self.model.vocab).replace(" EOS", "")
                results[(i * batch_size) + p] = {'ref': ref, 'greedy': arg_max, 'beam': '', 'text': ''}

        scores = [0, 0, 0]
        rouge_calc = RougeCalculator(stopwords=False, lang="en")

        for k in results:
            scores[0] += rouge_calc.rouge_1(results[k]['greedy'], results[k]['ref'])
            scores[1] += rouge_calc.rouge_2(results[k]['greedy'], results[k]['ref'])
            scores[2] += rouge_calc.rouge_l(results[k]['greedy'], results[k]['ref'])

        return scores[0] / len(results), scores[1] / len(results), scores[2] / len(results)

        
    