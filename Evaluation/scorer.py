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

class Scorer():
    def __init__(self, model):
        self.model = model
        self.sample_predictions = dict()
        self.test_samples = None
        self.beam_batch_size = 6



    def score_model(self, val_batches, use_cuda, beam, verbose=False):

        results = []

        for b in range(len(val_batches)):
            preds = self.model.predict_v2(val_batches[b], 120, beam, use_cuda)
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

                results[-1]['novelty_v2'] = pair.compute_novelty_v2(
                    [pair.get_tokens(text.split(" ")+["."], self.model.vocab)
                     for text in results[-1]['seq'].split(" . ")], self.model.vocab)

                results[-1]['ref_novelty'] = sum([sum(vec) / len(vec) for vec in pair.tri_gram_novelty_vector]) / \
                                                len(pair.tri_gram_novelty_vector)

            if b % 10 == 0 and verbose: print(b, ": ", len(val_batches))

        rouge_calc = RougeCalculator(stopwords=False, lang="en")
        scores = {"Rouge_1": 0, "Rouge_2": 0, "Rouge_L": 0, "Tri_novelty": 0, "p_gens": 0, "Tri_novelty_v2": 0}
        scores['rouge_dist'] = {}
        for rouge in ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-3-F', 'ROUGE-L-F']:
            scores['rouge_dist'][rouge] = []
            for k in range(21): scores['rouge_dist'][rouge].append([])

        if verbose: print("Computing SumEval scores")
        summaries, references = [], []

        for result in results:
            summaries.append(result['seq'])
            references.append(result['ref'])
            scores["Rouge_1"] += (rouge_calc.rouge_1(result['seq'], result['ref']) *100)
            scores["Rouge_2"] += (rouge_calc.rouge_2(result['seq'], result['ref']) *100)
            scores["Rouge_L"] += (rouge_calc.rouge_l(result['seq'], result['ref']) *100)
            scores["Tri_novelty"] += result['novelty']
            scores["Tri_novelty_v2"] += result['novelty_v2']
            if 'p_gen' in result: scores["p_gens"] += result['p_gen']

            perl_scores = self.score_rouge_org([result['seq']], [result['ref']])
            for rouge in scores['rouge_dist']:
                scores['rouge_dist'][rouge][int(20*result['ref_novelty'])].append(perl_scores[rouge])

        for k in scores:
            if k is 'rouge_dist':
                for rouge in scores['rouge_dist']:
                    scores['rouge_dist'][rouge] = [(sum(l)/len(l), len(l)) if len(l) > 0 else (0, 0) for l in scores['rouge_dist'][rouge]]

            else:
                scores[k] = scores[k] / len(results)

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