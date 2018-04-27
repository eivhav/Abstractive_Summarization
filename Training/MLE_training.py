from __future__ import unicode_literals, print_function, division


import Models.utils as utils
from Models.utils import *
from Models.pointer_gen import *
from Data.data_loader import *

from tensorboardX import SummaryWriter
from sumeval.metrics.rouge import RougeCalculator

class MLE_Trainer(): 
    def __init__(self, model, tag):
        self.model = model
        self.max_pool = nn.MaxPool2d((3, 1), stride=(1, 1))
        self.writer = SummaryWriter(comment=tag)


    def train(self, data_loader, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion,
              use_cuda, print_evry, novelty_loss=-1, start_iter=0):

        self.model.encoder_optimizer = optimizer(self.model.encoder.parameters(), lr= lr)
        self.model.decoder_optimizer = optimizer(self.model.decoder.parameters(), lr= lr)
        self.model.criterion = nn.NLLLoss()

        nb_training_samples = len(data_loader.manifest['training']['samples'].keys())
        self.model.logger = TrainingLogger(nb_epochs, batch_size, sample_size=nb_training_samples, val_size=1)

        for iter in range(start_iter, int(nb_epochs*nb_training_samples/batch_size)):

            batch = data_loader.sample_training_batch(batch_size)
            _time, losses = self.train_batch(batch, use_cuda=self.model.use_cuda, novelty_lambda=novelty_loss)
            for loss_type in losses:
                self.writer.add_scalar('1-Training/'+loss_type, losses[loss_type], iter)

            self.model.logger.add_iteration(iter, losses['loss_total'], _time)

            if iter % print_evry == 0:
                self.validate_model(data_loader, iter, batch_size, use_cuda, novelty_loss)
                if iter % (print_evry*3) == 0 and iter != 0:
                    self.model.save_model(self.model.config['model_path'], self.model.config['model_id'], epoch=iter, loss=0)



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
        if novelty_lambda > 0:
            scaled_att = (1 - p_gens) * previous_att

            _size = scaled_att.size()[2] - 3

            first_tri_gram = [scaled_att.narrow(1, i, 1).narrow(2, i, _size) for i in range(3)]
            novelty_att = torch.cat((first_tri_gram[0], first_tri_gram[1], first_tri_gram[2]), 1).unsqueeze(1)
            for t in range(1, target_length-2):
                tri_gram = [scaled_att.narrow(1, t+i, 1).narrow(2, i, _size) for i in range(3)]
                tri_sequence = torch.cat((tri_gram[0], tri_gram[1], tri_gram[2]), 1).unsqueeze(1)
                novelty_att = torch.cat((novelty_att, tri_sequence), 1)

            min_values = self.max_pool(-novelty_att).squeeze(2)
            novelty_loss = torch.abs(min_values.sum(-1).sum(-1).sum(-1))
            return_values['loss_novelty'] = novelty_loss.data[0] / target_length

            loss = mle_loss + (novelty_lambda * novelty_loss)

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



    def validate_model(self, data_loader, iter, batch_size, use_cuda, novelty_loss):

        val_batches = data_loader.load_data('val', batch_size)
        loss_values = dict()
        for batch in val_batches:
            _time, losses = self.train_batch(batch, use_cuda=self.model.use_cuda, backprop=False, novelty_lambda=novelty_loss)
            if len(loss_values) == 0:
                for k in losses: loss_values[k] = 0
            for k in losses: loss_values[k] += losses[k]
        for k in loss_values:
            self.writer.add_scalar('2-Validation/'+k, loss_values[k] / len(val_batches), iter)

        greedy_scores = self.score_model(val_batches[0:50], use_cuda, beam=False)
        for score_type in greedy_scores:
            self.writer.add_scalar('3-Greedy/'+ score_type, greedy_scores[score_type], iter)

        beam_scores = self.score_model([b[0:10] for b in val_batches[0:50]], use_cuda, beam=5)
        for score_type in beam_scores:
            self.writer.add_scalar('4-Beam/'+ score_type, beam_scores[score_type], iter)

        test_sample = data_loader.sample_training_batch(1)
        preds = self.model.predict(test_sample, self.model.config['target_length'], False, self.model.use_cuda)
        self.writer.add_text('Greedy Prediction', " ".join([t[0]['word'] for t in preds]), iter)

        preds_beam = self.model.predict(test_sample, self.model.config['target_length'], 5, self.model.use_cuda)
        self.writer.add_text('Beam Prediction', preds_beam[0][0], iter)



    def score_model(self, val_batches, use_cuda, beam):

        results = []

        for b in range(len(val_batches)):
            print(b)
            preds = self.model.predict_v2(val_batches[b], 75, beam, use_cuda)
            for p in range(len(val_batches[b])):
                pair = val_batches[b][p]
                ref = pair.get_text(pair.full_target_tokens, self.model.vocab).replace(" EOS", "")
                if not beam:
                    seq = [t[p] for t in preds]
                    results.append({'ref': ref, 'seq': " ".join([s['word'] for s in seq]).split(" EOS")[0]})
                    results[-1]['p_gen'] = sum([s['p_gen'] for s in seq[:len(results[-1]['seq'])]]) / len(results[-1]['seq'])
                    results[-1]['novelty'] = pair.compute_novelty([s['token_idx'] for s in seq[:len(results[-1]['seq'])]])
                else:
                    results.append({'ref': ref, 'seq': preds[p][0][0].split(" EOS")[0]})
                    results[-1]['novelty'] = pair.compute_novelty(
                        pair.get_tokens(results[-1]['seq'].split(" "), self.model.vocab))

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        scores = {"Rouge_1": 0, "Rouge_2": 0, "Rouge_L": 0, "Tri_novelty": 0}
        if not beam: scores["p_gens"] = 0

        summaries, references = [], []

        for result in results:
            summaries.append(result['seq'])
            references.append(result['ref'])
            scores["Rouge_1"] += (rouge_calc.rouge_1(result['seq'], result['ref']) *100)
            scores["Rouge_2"] += (rouge_calc.rouge_2(result['seq'], result['ref']) *100)
            scores["Rouge_L"] += (rouge_calc.rouge_l(result['seq'], result['ref']) *100)
            scores["Tri_novelty"] += result['novelty']
            if 'p_gen' in result: scores["p_gens"] += result['p_gen']

        for k in scores: scores[k] = scores[k] / len(results)

        return scores, self.score_rouge_org(summaries, references), results


    def score_rouge_org(self, sammaries, references):

        from pythonrouge.pythonrouge import Pythonrouge
        rouge = Pythonrouge(summary_file_exist=False,
                            summary=[s.replace(" . ", " .\n").split("\n") for s in sammaries],
                            reference=[[s.replace(" . ", " .\n").split("\n")] for s in references],
                            n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                            recall_only=False, stemming=True, stopwords=False,
                            word_level=True, length_limit=False, length=150,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=False, samples=1000, favor=True, p=0.5)
        return rouge.calc_score()





        
    