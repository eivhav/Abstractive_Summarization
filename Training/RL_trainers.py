
from __future__ import unicode_literals, print_function, division


import Models.utils as utils
from Models.utils import *
from Models.pointer_gen import *
from Data.data_loader import *
from Training.trainer import Trainer

from torch.distributions import *



class SelfCriticalTrainer(Trainer):
    def __init__(self, model, tag, rl_lambda, reward_module, reward_min):
        super().__init__(model, tag)

        self.rl_lambda = rl_lambda
        self.reward_module = reward_module
        self.reward_min = reward_min



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

        encoder_outputs, encoder_hidden = self.model.encoder(input_variable, encoder_hidden)
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1)
        decoder_h_states = torch.cat((encoder_hidden[0], encoder_hidden[1]), -1).unsqueeze(1)
        previous_att = None

        decoder_input = torch.cat((decoder_input, decoder_input, decoder_input), 0)
        decoder_h_states = torch.cat((decoder_h_states, decoder_h_states, decoder_h_states), 0)
        decoder_hidden = torch.cat((decoder_hidden, decoder_hidden, decoder_hidden), 0)
        encoder_outputs = torch.cat((encoder_outputs, encoder_outputs, encoder_outputs), 0)
        full_input_variable = torch.cat((full_input_variable, full_input_variable, full_input_variable), 0)
        control_var_zero = torch.cat((control_var_zero, control_var_zero, control_var_zero), 0)

        baseline_seq = [[] for s in range(len(samples))]
        sampled_seq = [[] for s in range(len(samples))]
        rl_losses = [0] * len(samples)
        mle_loss = 0
        p_gens = None

        for token_i in range(target_length):
            p_final, p_gen, p_vocab, att_dist, decoder_h_states, decoder_hidden, previous_att = \
                self.model.decoder(decoder_input, decoder_h_states, decoder_hidden, encoder_outputs,
                             full_input_variable, previous_att, control_var_zero, nb_unks, use_cuda)

            # First we compute the baseline sequence, argmax'ing over the output
            _, action_baseline = p_final.narrow(0, 0, len(samples)).max(1)

            # Next we sample of the p_vocab dist to get the agents action
            p_final_sampling = p_final.narrow(0, len(samples), len(samples))
            action_dist = Categorical(p_final_sampling)
            action_sampling = action_dist.sample()

            for i in range(len(samples)):
                baseline_seq[i].append(utils.translate_word(action_baseline.data[i], samples[i], self.model.vocab))
                sampled_seq[i].append(utils.translate_word(action_sampling.data[i], samples[i], self.model.vocab))

            # Add the difference between the output and the sampled tokens to the loss function

            for i in range(len(samples)):
                rl_losses[i] += self.model.criterion(torch.log(p_final_sampling[i].clamp(min=1e-8).unsqueeze(0)),
                                            action_sampling[i]) / len(samples)

            # Last we compute the MLE loss
            p_final_mle = p_final.narrow(0, 2*len(samples), len(samples))
            mle_loss += self.model.criterion(torch.log(p_final_mle.clamp(min=1e-8)),
                                             full_target_variable.narrow(1, token_i, 1).squeeze(-1))

            if random.uniform(0, 1) < tf_ratio: mle_decoder_input = target_variable.narrow(1, token_i, 1).squeeze(-1)
            else: _, mle_decoder_input = p_final_mle.max(1)

            decoder_input = torch.cat((self.mask_input(action_baseline),
                                       self.mask_input(action_sampling),
                                       self.mask_input(mle_decoder_input)), 0).unsqueeze(1)

            if token_i == 0: p_gens = p_gen.unsqueeze(-1)
            else: p_gens = torch.cat((p_gens, p_gen.unsqueeze(-1)), 1)

        rewards = self.reward_module.compute_reward(samples+samples, baseline_seq+sampled_seq, self.model)
        baseline_reward = rewards[0:len(samples)]
        sampled_reward = rewards[len(samples):]

        reward_delta = [max(self.reward_min, (sampled_reward[i] - baseline_reward[i]))
                        for i in range(len(baseline_reward))]

        rl_loss = 0
        for i in range(len(samples)):
            rl_loss += reward_delta[i] * rl_losses[i]
        joint_loss = (self.rl_lambda * rl_loss) + ((1- self.rl_lambda) * mle_loss)

        if backprop:
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm(self.model.decoder.parameters(), 2)
            self.model.encoder_optimizer.step()
            self.model.decoder_optimizer.step()

        return_values = dict()
        return_values['loss_total'] = joint_loss.data[0] / target_length
        return_values['loss_mle'] = (1- self.rl_lambda) * mle_loss.data[0] / target_length
        return_values['loss_rl'] = self.rl_lambda * rl_loss.data[0] / target_length
        return_values['rewards/avg_baseline'] = sum(baseline_reward) / len(samples)
        return_values['rewards/avg_sampled'] = sum(sampled_reward) / len(samples)
        return_values['rewards/avg_xdelta'] = sum(reward_delta) / len(samples)
        return_values['p_gens'] = p_gens.sum(-1).sum().data[0] / (target_length * len(samples) * 3)

        return time.time() - start, return_values
















class MonteCarloTrainer(Trainer):
    def __init__(self, model, tag):
        super().__init__(model, tag)


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
                self.model.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs,
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
        source_texts = [" ".join(pair.get_text(pair.full_target_tokens,
                                               self.model.vocab).split("EOS")[0].split(" ")[:25]) for pair in pairs]
        scores = []
        for s in range(len(sequences)):
            generated_texts = [pairs[s].get_text(seq, self.model.vocab).split("EOS")[0] for seq in sequences[s]]
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

        target_length = min(self.model.config['target_length'], max([len(pair.masked_target_tokens) for pair in samples]))

        nb_unks = max([len(s.unknown_tokens) for s in samples])
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
            utils.get_batch_variables(samples, self.model.config['input_length'], target_length, use_cuda,
                                      self.model.vocab.word2index['SOS'])

        encoder_hidden = self.model.encoder.init_hidden(len(samples), use_cuda)
        self.model.encoder_optimizer.zero_grad()
        self.model.decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.model.encoder(input_variable, encoder_hidden)
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
                self.model.decoder(decoder_input, decoder_hidden_states, decoder_hidden, encoder_outputs,
                             full_input_variable, previous_att, nb_unks, use_cuda)

            action_dist = Categorical(p_final)
            action_sampling = action_dist.sample()

            for s in range(len(samples)): result_seq[s].append(action_sampling.data[s])
            decoder_input = self.model.mask_input(action_sampling).unsqueeze(1)

            timings['decoding&max'] += time.time() - last
            last = time.time()
            if token_i == target_length -1:
                rewards = self.model.compute_reward_mc(samples, [[seq] for seq in result_seq], rouge)
                timings['reward'] += time.time() - last
                last = time.time()

            else:
                mc_seqs = self.model.run_N_monte_carlo_simulations(1, len(samples), decoder_input.clone(),
                                                         decoder_hidden_states.clone(), decoder_hidden.clone(),
                                                         encoder_outputs, full_input_variable, previous_att.clone(),
                                                         nb_unks, use_cuda, sim_length=target_length-token_i -1)
                timings['mc'] += time.time() -last
                last = time.time()
                reward_seqs = [[result_seq[s] + mc_seqs[s][n] for n in range(len(mc_seqs[s]))] for s in range(len(samples))]

                rewards = self.model.compute_reward_mc(samples, reward_seqs, rouge)
                timings['reward'] += time.time() - last
                last = time.time()

            # reward scaling?
            #print(token_i, sum(rewards) / len(rewards))
            losses.append([rewards,
                    [self.model.criterion(torch.log(p_final[i].clamp(min=1e-8).unsqueeze(0)), action_sampling[i])
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
            decoder_input = self.model.mask_input(max_tokens).unsqueeze(1)

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
            torch.nn.utils.clip_grad_norm(self.model.encoder.parameters(), 2)
            torch.nn.utils.clip_grad_norm(self.model.decoder.parameters(), 2)
            self.model.encoder_optimizer.step()
            self.model.decoder_optimizer.step()
        else:
            print("No update")
            return 0, time.time() - start
        timings['backprop'] += time.time() - last

        return loss.data[0] / target_length, time.time() - start, baseline_rewards


        
        


