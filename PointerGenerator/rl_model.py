
from torch.distributions import *

from PointerGenerator.model import *
from sumeval.metrics.rouge import RougeCalculator

class PGmodel_reinforcement(PGModel):
    def __init__(self, config, vocab, use_cuda):
        super().__init__(config, vocab, use_cuda)


    def train_rl(self, data, val_data, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion, use_cuda, print_evry):

        if self.logger is None:
            self.encoder_optimizer = optimizer(self.encoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.decoder_optimizer = optimizer(self.decoder.parameters(), lr= lr, weight_decay=0.0000001)
            self.criterion = nn.NLLLoss()
            self.logger = TrainingLogger(nb_epochs, batch_size, len(data), len(val_data))
            print("Optimizers compiled for RL training")

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        for epoch in range(len(self.logger.log), nb_epochs):
            self.logger.init_epoch(epoch)
            batches = utils.sort_and_shuffle_data(data, nb_buckets=100, batch_size=batch_size, rnd=True)
            rouge = 0
            for b in range(len(batches)):
                '''
                print("\n")
                preds = self.predict(batches[b], self.config['target_length'], False, self.use_cuda)
                rewards = self.compute_reward_mc(batches[b],
                                                 [[[t[s]['token_idx'] for t in preds]] for s in range(batch_size)],
                                                 rouge_calc)
                score_before = sum(rewards) / len(rewards)

                #b_p = [[p.clone() for p in self.encoder.parameters()], [p.clone() for p in self.decoder.parameters()]]
                 '''
                rewards = [0 for s in batches[b]]
                loss, _time, base_r = self.train_batch_rl_mc(samples=batches[b], use_cuda=self.use_cuda, rouge=rouge_calc)

                rouge += sum(base_r) / len(base_r)

                '''
                a_p = [[p for p in self.encoder.parameters()], [p for p in self.decoder.parameters()]]
                print("encoder", [torch.equal(b_p[0][i].data, a_p[0][i].data) for i in range(len(a_p[0]))])
                print("decoder", [torch.equal(b_p[1][i].data, a_p[1][i].data) for i in range(len(a_p[1]))])
                

                preds = self.predict(batches[b], self.config['target_length'], False, self.use_cuda)
                rewards = self.compute_reward_mc(batches[b],
                                                 [[[t[s]['token_idx'] for t in preds]] for s in range(batch_size)],
                                                 rouge_calc)
                score_after = sum(rewards) / len(rewards)
                print('diff', round((score_after - score_before), 3))
                 '''
                self.logger.add_iteration(b+1, loss, _time)

                if b % print_evry == 0:
                    preds = self.predict([data[b*batch_size]], self.config['target_length'], False, self.use_cuda)
                    print('\n', " ".join([str(t[0]['word']) for t in preds]))
                    print("rouge [1, 2, l]", self.test_rouge_scores(val_data[0:200], batch_size=20, beam=False,
                                                            rouge_calc=rouge_calc),
                                            "training_rouge", round(rouge / print_evry, 4))
                    rouge = 0


            for b in range(int(len(val_data)/batch_size)):
                try:
                    loss, _time = self.train_batch(val_data[b*batch_size:(b+1)*batch_size], self.use_cuda, backprop=False)
                    self.logger.add_val_iteration(b+1, loss, _time)

                except:
                    print("\n", "Error during validation!")

            print("rouge for epoch [1, 2, l]", self.test_rouge_scores(val_data[0:5000], batch_size=20, beam=False,
                                                                      rouge_calc=rouge_calc))

            if True or epoch == 0 or self.logger.log[epoch]["val_loss"] < self.logger.log[epoch-1]["val_loss"]:
                self.save_model(self.config['model_path'], self.config['model_id'],
                                epoch=epoch, loss=self.logger.log[epoch]["val_loss"])


    def test_rouge_scores(self, data, batch_size, beam, rouge_calc):
        results = [[], [], []]
        rouges = [rouge_calc.rouge_1, rouge_calc.rouge_2, rouge_calc.rouge_l]
        if not beam:
            for b in range(int(len(data) / batch_size)):
                seqs = self.predict(data[b*batch_size:(b+1)*batch_size], self.config['target_length'], False, self.use_cuda)
                for i in range(3):
                    results[i] += self.compute_reward(data[b*batch_size:(b+1)*batch_size], seqs, rouges[i])

        return [round(sum(r) / len(r), 3) for r in results]



    def run_N_monte_carlo_simulations(self, N, batch_size, decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs,
                                      full_input_variable, previous_att, nb_unks, use_cuda, sim_length):

        def stack_N(tensor):
            return torch.stack([tensor[i] for i in range(batch_size) for n in range(N)], 0)

        decoder_input = stack_N(decoder_input)
        decoder_hidden_states = stack_N(decoder_hidden_states)
        decoder_hidden = stack_N(decoder_hidden)
        encoder_outputs = stack_N(encoder_outputs)
        full_input_variable = stack_N(full_input_variable)
        previous_att = stack_N(previous_att)

        sampled_seq = []
        for i in range(batch_size):
            sim_seq = []
            for i in range(N): sim_seq.append([])
            sampled_seq.append(sim_seq)

        for token_i in range(sim_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs,
                             full_input_variable, previous_att, nb_unks, use_cuda)

            _, max_tokens = p_final.max(1)
            action_sampling = max_tokens
            #action_dist = Categorical(p_final)
            #action_sampling = action_dist.sample()

            for i in range(action_sampling.size()[0]):
                sampled_seq[int(i/N)][i % N].append(action_sampling.data[i])
            decoder_input = self.mask_input(action_sampling).unsqueeze(1)

        return sampled_seq

    def compute_reward_mc(self, pairs, sequences, rouge):
        # return [1 for i in range(len(pairs))]
        source_texts = [" ".join(pair.get_text(pair.full_target_tokens, self.vocab).split("EOS")[0].split(" ")[:25]) for pair in pairs]
        scores = []
        for s in range(len(sequences)):
            generated_texts = [pairs[s].get_text(seq, self.vocab).split("EOS")[0] for seq in sequences[s]]
            #print(generated_texts[0])
            rouge_scores = [rouge.rouge_l(text, source_texts[s]) for text in generated_texts]
            #rouge_scores = [1 - (len(text.split(" ")) / 30)  for text in generated_texts]
            if len(rouge_scores) != 0: scores.append(sum(rouge_scores) / len(rouge_scores))
            else: scores.append(0)
        return scores



    def train_batch_rl_mc(self, samples, use_cuda, tf_ratio=0.5, backprop=True, coverage_lambda=-1, rouge=None, b_rewards=None):
        start = time.time()
        if len(samples) == 0: return 0, 0
        timings = {'var_init': 0, 'decoding&max': 0, 'reward': 0, 'mc': 0,  'loss': 0, 'backprop': 0}

        target_length = min(self.config['target_length'], max([len(pair.masked_target_tokens) for pair in samples]))
        #print("\nTarget_length", target_length)
        nb_unks = max([len(s.unknown_tokens) for s in samples])
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.config['input_length'], target_length, use_cuda,
                                      self.vocab.word2index['SOS'])

        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()


        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_hidden_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        result_seq = []
        for b in range(len(samples)): result_seq.append([])
        timings['var_init'] += time.time() - start
        last = time.time()

        losses = []
        for token_i in range(target_length):

            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs,
                             full_input_variable, previous_att, nb_unks, use_cuda)

            action_dist = Categorical(p_final)
            action_sampling = action_dist.sample()

            for s in range(len(samples)): result_seq[s].append(action_sampling.data[s])
            decoder_input = self.mask_input(action_sampling).unsqueeze(1)

            timings['decoding&max'] += time.time() - last
            last = time.time()
            if token_i == target_length -1:
                rewards = self.compute_reward_mc(samples, [[seq] for seq in result_seq], rouge)
                timings['reward'] += time.time() - last
                last = time.time()

            else:
                mc_seqs = self.run_N_monte_carlo_simulations(1, len(samples), decoder_input.clone(),
                                                         decoder_hidden_states.clone(), decoder_hidden.clone(),
                                                         encoder_outputs, full_input_variable, previous_att.clone(),
                                                         nb_unks, use_cuda, sim_length=target_length-token_i -1)
                timings['mc'] += time.time() -last
                last = time.time()
                reward_seqs = [[result_seq[s] + mc_seqs[s][n] for n in range(len(mc_seqs[s]))] for s in range(len(samples))]

                rewards = self.compute_reward_mc(samples, reward_seqs, rouge)
                timings['reward'] += time.time() - last
                last = time.time()

            # reward scaling?
            #print(token_i, sum(rewards) / len(rewards))
            losses.append([rewards,
                    [self.criterion(torch.log(p_final[i].clamp(min=1e-8).unsqueeze(0)), action_sampling[i])
                     for i in range(len(samples))]])
            '''
            for i in range(len(samples)):
                #print(b_rewards[i], rewards[i], max(0, rewards[i] - b_rewards[i]))
                if rewards[i] - b_rewards[i] > 0:
                    losses[i] += (rewards[i] - b_rewards[i]) \
                             * 
            '''
            timings['loss'] += time.time() - last
            last = time.time()

            _, max_tokens = p_final.max(1)
            for s in range(len(samples)): result_seq[s][-1] = max_tokens.data[s]
            decoder_input = self.mask_input(max_tokens).unsqueeze(1)

        baseline_rewards = losses[-1][0]
        loss = 0
        for timestep_loss in losses:
            #print(timestep_loss[0][0], baseline_rewards[0])
            for s in range(len(timestep_loss[0])):
                loss += max(0, (timestep_loss[0][s] - baseline_rewards[s])) * timestep_loss[1][s]

        timings['loss'] += time.time() - last
        last = time.time()

        if backprop and not isinstance(loss, int):
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 2)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        else:
            print("No update")
            return 0, time.time() - start
        timings['backprop'] += time.time() - last
        '''
        print("\n")
        for desc in timings: print(desc, round(timings[desc], 3), round(timings[desc] / sum([timings[t] for t in timings]), 2))
        '''

        return loss.data[0] / target_length, time.time() - start, baseline_rewards




    def train_batch_rl(self, samples, use_cuda, tf_ratio=0.5, backprop=True, coverage_lambda=-1,
                       rouge=None, b_rewards=None, loss_lambda=0.5):
        start = time.time()
        if len(samples) == 0: return 0, 0

        target_length = min(self.config['target_length'], max([len(pair.masked_target_tokens) for pair in samples]))
        nb_unks = max([len(s.unknown_tokens) for s in samples])

        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.config['input_length'], target_length, use_cuda,
                                      self.vocab.word2index['SOS'])

        encoder_hidden = self.encoder.init_hidden(len(samples), use_cuda)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_h_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        decoder_input = torch.cat((decoder_input, torch.cat((decoder_input, decoder_input), 0)), 0)
        decoder_h_states = torch.cat((decoder_h_states, torch.cat((decoder_h_states, decoder_h_states), 0)), 0)
        decoder_hidden = torch.cat((decoder_hidden, torch.cat((decoder_hidden, decoder_hidden), 0)), 0)
        encoder_outputs = torch.cat((encoder_outputs, torch.cat((encoder_outputs, encoder_outputs), 0)), 0)
        full_input_variable = torch.cat((full_input_variable, torch.cat((full_input_variable, full_input_variable), 0)), 0)

        baseline_seq = []
        sampled_seq = []
        rl_losses = [0] * len(samples)
        mle_loss = 0
        for token_i in range(target_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs,
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
                rl_losses[i] += self.criterion(torch.log(p_final_sampling[i].clamp(min=1e-8).unsqueeze(0)),
                                            action_sampling[i])

            # Last we compute the MLE loss
            p_final_mle = p_final.narrow(0, 2*len(samples), len(samples))
            mle_loss += self.criterion(torch.log(p_final_mle.clamp(min=1e-8)), full_target_variable.narrow(1, token_i, 1)
                                   .squeeze(-1))

            if random.uniform(0, 1) < tf_ratio: mle_decoder_input = target_variable.narrow(1, token_i, 1).squeeze(-1)
            else: _, mle_decoder_input = p_final_mle.max(1)

            decoder_input = torch.cat((torch.cat((self.mask_input(action_baseline), self.mask_input(action_sampling)), 0),
                                       self.mask_input(mle_decoder_input)), 0).unsqueeze(1)

        baseline_reward = self.compute_reward(samples, baseline_seq, rouge)
        sampled_reward = self.compute_reward(samples, sampled_seq, rouge)
        reward_delta = [max(-1, (sampled_reward[i] - baseline_reward[i])) for i in range(len(baseline_reward))]
        #print("\n reward percent", sum([1 / len(reward_delta) for r in reward_delta if r != 0]))

        rl_loss = 0
        for i in range(len(samples)):
            rl_loss += reward_delta[i] * rl_losses[i]
        joint_loss = (loss_lambda * rl_loss) + ((1- loss_lambda) * mle_loss)

        if backprop:
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 2)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        return joint_loss.data[0] / target_length, time.time() - start, baseline_reward


    def mask_input(self, input_var):
        for i in range(input_var.size()[0]):
            if input_var.data[i] >= self.vocab.vocab_size: input_var.data[i] = self.vocab.word2index['UNK']
        return input_var



    def compute_reward(self, pairs, sequences, rouge):

        source_texts = [pair.get_text(pair.full_target_tokens, self.vocab) for pair in pairs]
        generated_texts = [" ".join([str(sequences[w][i]['word']) for w in range(len(sequences))]) for i in range(len(pairs))]
        scores = [rouge(text[0], text[1]) for text in
                  [(generated_texts[i], source_texts[i]) for i in range(len(pairs))]]
        #print(scores[0], generated_texts[0][:100])
        ##print(scores[1], generated_texts[1][:100])
        #print(scores[2], generated_texts[2][:100])
        #print()
        return scores

