from __future__ import unicode_literals, print_function, division


import Models.utils as utils
from Models.pointer_gen import *
from Data.data_loader import *
from Training.trainer import Trainer


class MLE_Novelty_Trainer(Trainer):
    def __init__(self, model, tag, novelty_lambda):
        super().__init__(model, tag)

        self.novelty_lambda = novelty_lambda
        self.novelty_loss_type = 'combo'
        self.novelty_n_grams = 3
        self.max_pool = nn.MaxPool2d((self.novelty_n_grams, 1), stride=(1, 1))
        self.max_pool_1d = nn.MaxPool1d(self.novelty_n_grams, stride=1)


    def train_batch(self, samples, use_cuda, tf_ratio=0.5, backprop=True):
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
        mle_loss = 0

        encoder_outputs, encoder_hidden = self.model.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_h_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None
        p_gens = None
        novelty_tokens = [0] * len(samples)

        for token_i in range(target_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.model.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs,full_input_variable,
                                   previous_att, control_var_zero, nb_unks, use_cuda)

            mle_loss += self.model.criterion(torch.log(p_final.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1)
                                       .squeeze(-1))

            if random.uniform(0, 1) < tf_ratio: decoder_input = target_variable.narrow(1, token_i, 1)
            else:
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if max_tokens.data[i] >= self.model.vocab.vocab_size:
                        max_tokens.data[i] = self.model.vocab.word2index['UNK']
                decoder_input = max_tokens.unsqueeze(1)

            if token_i == 0: p_gens = p_gen.unsqueeze(-1)
            else: p_gens = torch.cat((p_gens, p_gen.unsqueeze(-1)), 1)

            if token_i >= 2 :
                _, max_tokens = p_final.max(1)
                for i in range(max_tokens.size()[0]):
                    if token_i <= len(samples[i].full_target_tokens):
                        last_tokens = (samples[i].full_target_tokens[token_i-2], samples[i].full_target_tokens[token_i-1])
                        last_tri = str(last_tokens[0]) + "~" + str(last_tokens[1]) + "~" + str(max_tokens.data[i])
                        if last_tri not in samples[i].source_tri_grams: novelty_tokens[i] += 1

        return_values = dict()
        if self.novelty_lambda > 0:
            if self.novelty_loss_type == 'p_gen':
                p_copy = (1 - p_gens).squeeze(-1)
                tri_grams = [p_copy.narrow(1, i, self.novelty_n_grams).unsqueeze(1)
                             for i in range(0, target_length-self.novelty_n_grams+1)]
                novelty_p_copy = torch.cat(tri_grams, 1)
                min_values = self.max_pool_1d(-novelty_p_copy).squeeze(-1)
                novelty_loss = torch.abs(min_values.sum(-1).sum(-1))
                return_values['loss_novelty'] = novelty_loss.data[0] / target_length
                loss = mle_loss + (self.novelty_lambda * novelty_loss)

            else:
                if self.novelty_loss_type == 'combo':
                    scaled_att = (1 - p_gens) * previous_att
                else:
                    scaled_att = previous_att

                _size = scaled_att.size()[2] - self.novelty_n_grams
                first_tri_gram = [scaled_att.narrow(1, i, 1).narrow(2, i, _size) for i in range(self.novelty_n_grams)]
                novelty_att = torch.cat(first_tri_gram, 1).unsqueeze(1)
                for t in range(1, target_length-self.novelty_n_grams+1):
                    tri_gram = [scaled_att.narrow(1, t+i, 1).narrow(2, i, _size) for i in range(self.novelty_n_grams)]
                    tri_sequence = torch.cat(tri_gram, 1).unsqueeze(1)
                    novelty_att = torch.cat((novelty_att, tri_sequence), 1)

                #return_values['novelty_att'] = novelty_att
                min_values = self.max_pool(-novelty_att).squeeze(2)
                #return_values['novelty_att'] = min_values
                #return_values['n_loss_vec'] = torch.abs(min_values.sum(-1).squeeze(0))
                novelty_loss = torch.abs(min_values.sum(-1).sum(-1).sum(-1))

                return_values['loss_novelty'] = novelty_loss.data[0] / target_length

                loss = mle_loss + (self.novelty_lambda * novelty_loss)

        else:
            loss = mle_loss

        if backprop:
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm(self.model.decoder.parameters(), 2)
            self.model.encoder_optimizer.step()
            self.model.decoder_optimizer.step()

        return_values['loss_total'] = loss.data[0] / target_length
        return_values['loss_mle'] = mle_loss.data[0] / target_length
        return_values['p_gens'] = p_gens.sum(-1).sum().data[0] / (target_length * len(samples))
        return_values['novelty_tokens'] = sum(novelty_tokens) / ((target_length-2) * len(samples))

        return time.time() - start, return_values