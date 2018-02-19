
from torch.distributions import *

from PointerGenerator.model import *

class PGmodel_reinforcement(PGModel):
    def __init__(self, config, vocab, use_cuda):
        super().__init__(config, vocab, use_cuda)


    def train_rl(self, data, val_data, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion, use_cuda, print_evry):

        if self.logger is None:
            self.encoder_optimizer = optimizer(self.encoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.decoder_optimizer = optimizer(self.decoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.criterion = nn.NLLLoss()
            self.logger = TrainingLogger(nb_epochs, batch_size, len(data), len(val_data))
            print("Optimizers compiled")

        for epoch in range(len(self.logger.log), nb_epochs):
            self.logger.init_epoch(epoch)
            batches = utils.sort_and_shuffle_data(data, nb_buckets=100, batch_size=batch_size, rnd=False)
            for b in range(len(batches)):
                #try:
                loss, _time = self.train_batch_rl(samples=batches[b], use_cuda=self.use_cuda)
                self.logger.add_iteration(b+1, loss, _time)
                if b % print_evry == 0:
                    preds = self.predict([data[b*batch_size]], self.config['target_length'], False, self.use_cuda)
                    print('\n', " ".join([t[0]['word'] for t in preds]))
                #except:
                    #print("\n", "Error for batch ", b, " size:", len(batches[b]))
            for b in range(int(len(val_data)/batch_size)):
                try:
                    loss, _time = self.train_batch(val_data[b*batch_size:(b+1)*batch_size], self.use_cuda, backprop=False)
                    self.logger.add_val_iteration(b+1, loss, _time)
                except:
                    print("\n", "Error during validation!")

            if epoch == 0 or self.logger.log[epoch]["val_loss"] < self.logger.log[epoch-1]["val_loss"]:
                self.save_model(self.config['model_path'], self.config['model_id'],
                                epoch=epoch, loss=self.logger.log[epoch]["val_loss"])



    def train_batch_rl(self, samples, use_cuda, tf_ratio=0.5, backprop=True, coverage_lambda=-1):
        start = time.time()
        if len(samples) == 0: return 0, 0
        print("Training rL")

        target_length = min(self.config['target_length'], max([len(pair.masked_target_tokens) for pair in samples]))
        nb_unks = max([len(s.unknown_tokens) for s in samples])

        print(target_length, nb_unks)
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.config['input_length'], target_length, use_cuda,
                                      self.vocab.word2index['SOS'])

        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        losses = [0] * len(samples)

        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_hidden_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        decoder_input = torch.cat((decoder_input, decoder_input), 0)
        decoder_hidden_states = torch.cat((decoder_hidden_states, decoder_hidden_states), 0)
        decoder_hidden = torch.cat((decoder_hidden, decoder_hidden), 0)
        encoder_outputs = torch.cat((encoder_outputs, encoder_outputs), 0)
        full_input_variable = torch.cat((full_input_variable, full_input_variable), 0)

        baseline_seq = []
        sampled_seq = []
        for token_i in range(target_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs,
                             full_input_variable, previous_att, nb_unks, use_cuda)

            # First we compute the baseline sequence, argmax'ing over the output
            _, action_baseline = p_final.narrow(0, 0, len(samples)).max(1)
            baseline_seq.append([{'token_idx': action_baseline.data[i],
                                  'word': utils.translate_word(action_baseline.data[i], samples[i], self.vocab)
                                } for i in range(len(samples))])

            # Next we sample of the p_vocab dist to get the agents action
            p_final_sampling = p_final.narrow(0, len(samples), len(samples))
            action_dist = Categorical(p_final_sampling)
            action_sampling = action_dist.sample()
            sampled_seq.append([{'token_idx': action_sampling.data[i],
                                 'word': utils.translate_word(action_sampling.data[i], samples[i], self.vocab)
                                } for i in range(len(samples))])

            # Add the difference between the output and the sampled tokens to the loss function

            for i in range(len(samples)):
                losses[i] += self.criterion(torch.log(p_final_sampling[i].clamp(min=1e-8).unsqueeze(0)),
                                            action_sampling[i])

            decoder_input = torch.cat((action_baseline, action_sampling), 0).unsqueeze(1)

        reward_baseline = [self.compute_reward(samples[i], [baseline_seq[w][i] for w in range(len(baseline_seq))])
                           for i in range(len(samples))]
        reward_sampled = [self.compute_reward(samples[i], [sampled_seq[w][i] for w in range(len(sampled_seq))])
                           for i in range(len(samples))]
        reward_delta = torch.FloatTensor(reward_sampled) + torch.FloatTensor(reward_baseline)
        reward_delta = Variable(reward_delta, requires_grad=True)
        if use_cuda: reward_delta = reward_delta.cuda()
        loss = 0
        for i in range(len(samples)):
            loss += reward_delta[i] * losses[i]

        if backprop:
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        print(time.time() - start)
        return loss.data[0] / target_length, time.time() - start


    def compute_reward(self, source_pair, sentence):
        return 0.5