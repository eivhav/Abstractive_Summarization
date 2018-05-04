from __future__ import unicode_literals, print_function, division
import json

import Models.utils as utils
from Models.utils import *
from Models.pointer_gen import *
from Data.data_loader import *
import torch

from tensorboardX import SummaryWriter
from sumeval.metrics.rouge import RougeCalculator
from pythonrouge.pythonrouge import Pythonrouge

class Trainer():
    def __init__(self, model, tag):
        self.model = model
        self.max_pool = nn.MaxPool2d((3, 1), stride=(1, 1))
        self.writer = SummaryWriter(comment=tag)
        self.sample_predictions = dict()
        self.test_samples = None



    def train(self, data_loader, nb_epochs, batch_size, optimizer, lr, tf_ratio, stop_criterion,
              use_cuda, print_evry, novelty_loss=-1, start_iter=0, new_optm=True):


        if new_optm or self.model.encoder_optimizer is None:
            self.model.encoder_optimizer = optimizer(self.model.encoder.parameters(), lr= lr)
            self.model.decoder_optimizer = optimizer(self.model.decoder.parameters(), lr= lr)
        self.model.criterion = nn.NLLLoss()

        nb_training_samples = len(data_loader.manifest['training']['samples'].keys())
        self.model.logger = TrainingLogger(nb_epochs, batch_size, sample_size=nb_training_samples, val_size=1)

        for iter in range(start_iter, int(nb_epochs*nb_training_samples/batch_size)):

            batch = data_loader.sample_training_batch(batch_size)
            _time, losses = self.train_batch(batch, use_cuda=self.model.use_cuda)
            for loss_type in losses:
                self.writer.add_scalar('1-Training/'+loss_type, losses[loss_type], iter)

            self.model.logger.add_iteration(iter, losses['loss_total'], _time)

            if iter % print_evry == 0:
                self.validate_model(data_loader, iter, batch_size, use_cuda, novelty_loss)
                if iter % print_evry == 0 and iter != 0:
                    self.model.save_model(self.model.config['model_path'], self.model.config['model_id'], epoch=iter, loss=0)




    def validate_model(self, data_loader, iter, batch_size, use_cuda, novelty_loss):

        val_batches = data_loader.load_data('val', batch_size)
        loss_values = dict()
        for batch in val_batches[:50]:
            _time, losses = self.train_batch(batch, use_cuda=self.model.use_cuda, backprop=False)
            if len(loss_values) == 0:
                for k in losses: loss_values[k] = 0
            for k in losses: loss_values[k] += losses[k]
        for k in loss_values:
            self.writer.add_scalar('2-Validation/'+k, loss_values[k] / len(val_batches[:50]), iter)

        greedy_scores = self.score_model(val_batches[0:50], use_cuda, beam=False)
        for score_type in greedy_scores:
            self.writer.add_scalar('3-Greedy/'+ score_type, greedy_scores[score_type], iter)

        beam_scores = self.score_model([b[0:10] for b in val_batches[0:50]], use_cuda, beam=5)
        for score_type in beam_scores:
            self.writer.add_scalar('4-Beam/'+ score_type, beam_scores[score_type], iter)

        if len(self.sample_predictions) < 5:
            self.test_samples = data_loader.sample_training_batch(5)
            for sample in self.test_samples:
                self.sample_predictions[sample.id] = {'1source:': " ".join(sample.source[:400]).split(" . "),
                                                      '2target': [" ".join(t) for t in sample.target],
                                                      '3beam': dict(), '4greedy': dict()
                                                      }

        preds = self.model.predict_v2(self.test_samples, self.model.config['target_length'], False, self.model.use_cuda)
        preds_beam = self.model.predict_v2(self.test_samples, self.model.config['target_length'], 5, self.model.use_cuda)
        for i in range(len(self.test_samples)):
            self.sample_predictions[self.test_samples[i].id]['3beam'][iter] = preds_beam[i][0][0]
            self.sample_predictions[self.test_samples[i].id]['4greedy'][iter] = " ".join([t[i]['word'] for t in preds])

        log_dir = self.writer.file_writer.get_logdir()
        with open(log_dir+'/predictions.json', 'w') as outfile: json.dump(self.sample_predictions, outfile)



    def score_model(self, val_batches, use_cuda, beam, verbose=False):

        results = []

        for b in range(len(val_batches)):
            preds = self.model.predict_v2(val_batches[b], 100, beam, use_cuda)
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
                    results[-1]['p_gen'] = preds[p][0][2]
                    results[-1]['novelty'] = pair.compute_novelty(
                        pair.get_tokens(results[-1]['seq'].split(" "), self.model.vocab))

            if b % 10 == 0 and verbose: print(b, ": ", len(val_batches))

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        scores = {"Rouge_1": 0, "Rouge_2": 0, "Rouge_L": 0, "Tri_novelty": 0, "p_gens": 0}

        if verbose: print("Computing SumEval scores")
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

        if verbose: print("Computing Perl scores")
        perl_scores = self.score_rouge_org(summaries, references)
        for k in perl_scores:
            if k[-1] != "R":
                id = "R" + k.replace("-", "_").lower()[1:-1] + "perl"
                scores[id] = 100*perl_scores[k]

        return scores


    def score_rouge_org(self, sammaries, references):

        rouge = Pythonrouge(summary_file_exist=False,
                            summary=[s.replace(" . ", " .\n").split("\n") for s in sammaries],
                            reference=[[s.replace(" . ", " .\n").split("\n")] for s in references],
                            n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                            recall_only=False, stemming=True, stopwords=False,
                            word_level=True, length_limit=False, length=150,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=False, samples=1000, favor=True, p=0.5)
        return rouge.calc_score()

    def mask_input(self, input_var, UNK_vector=None):
        if UNK_vector is not None:
            mask = input_var.ge(self.model.vocab.vocab_size).long()
            return ((1 - mask) * input_var) + (mask * UNK_vector)

        for i in range(input_var.size()[0]):
            if input_var.data[i] >= self.model.vocab.vocab_size:
                input_var.data[i] = self.model.vocab.word2index['UNK']
        return input_var


