from pythonrouge.pythonrouge import Pythonrouge

from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
import time
from multiprocessing import Process, Queue
import random
#from threading import Thread
#from queue import Queue


class AlternateRewards:
    def __init__(self, rewards):
        self.rewards = rewards

    def compute_reward(self, samples, sequence, model):
        reward_module = random.choice(self.rewards)
        return reward_module.compute_reward(samples, sequence, model)


class ComboRewards:
    def __init__(self, rewards, weights):
        self.rewards = rewards
        self.weights = weights

    def compute_reward(self, samples, sequence, model):
        reward_lists = [r.compute_reward(samples, sequence, model) for r in self.rewards]
        weighted_rewards = [[r*self.weights[i] for r in reward_lists[i]] for i in range(len(reward_lists))]
        return [sum([reward[s] for reward in weighted_rewards]) for s in range(len(samples))]


class RougeWorkers():
    def __init__(self, nb, result_queue):
        self.nb = nb
        self.q = Queue()
        self.result_queue = result_queue

    def compute_perl_scores(self, reference, summary):
        rouge = Pythonrouge(summary_file_exist=False,
                            summary=[s.replace(" . ", " .\n").split("\n") for s in [summary]],
                            reference=[[s.replace(" . ", " .\n").split("\n")] for s in [reference]],
                            n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                            recall_only=False, stemming=True, stopwords=False,
                            word_level=True, length_limit=False, length=150,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=False, samples=1000, favor=True, p=0.5)
        return rouge.calc_score()

    def serve(self):
        while True:
            order = self.q.get()
            id = order[2]
            score = self.compute_perl_scores(order[0], order[1])
            self.result_queue.put((score, id))

    def start_worker(self):
        p = Process(target=self.serve)
        p.daemon = True
        p.start()


class RougePerlVersion():

    def __init__(self, rouge_variants, nb_workers=4):
        self.rouge_variants = rouge_variants
        self.score_queue = Queue()
        self.workers = [RougeWorkers(i, self.score_queue) for i in range(nb_workers)]
        for w in self.workers: w.start_worker()


    def compute_reward(self, samples, sequence, model):

        references = [pair.get_text(pair.full_target_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        summaries = [" ".join([str(token) for token in s]).split(" EOS")[0] for s in sequence]

        start = time.time()

        for i in range(len(summaries)):
            self.workers[i % len(self.workers)].q.put((references[i], summaries[i], i))

        results = [self.score_queue.get() for i in range(len(summaries))]
        scores = {r[1]: r[0] for r in results}

        final_scores = []
        for i in range(len(summaries)):
            final_scores.append(sum([float(scores[i][rouge]) for rouge in self.rouge_variants if rouge in scores[i]]) \
                                / len(self.rouge_variants))

        return final_scores


class TrigramNovelty:
    def __init__(self, remove_stopwords=False, stem=False, method='tri_gram'):
        self.rouge_calc = RougeCalculator(stopwords=remove_stopwords, lang="en", stemming=stem)

        if method =='tri_gram': self.compute_n_grams = self.compute_tri_grams
        elif method == 'bi_gram': self.compute_n_grams = self.compute_bi_grams
        else: print("Invalid n-gram specifier", method)


    def compute_tri_grams(self, tokens):
        #print(tokens)
        return {str(tokens[i]) + "~" + str(tokens[i + 1]) + "~" + str(tokens[i + 2]): True for i in
                         range(len(tokens) - 2)}

    def compute_bi_grams(self, tokens):
        return {str(tokens[i]) + "~" + str(tokens[i + 1]): True for i in range(len(tokens) - 1)}


    def compute_reward(self, samples, sequence, model, return_grams=False):

        sources = [pair.get_text(pair.full_source_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        references = [pair.get_text(pair.full_target_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        summaries = [" ".join([str(token) for token in s]).split(" EOS")[0] for s in sequence]

        scores = []
        reward_grams = []
        for i in range(len(references)):
            hyps_tri_grams = [self.compute_n_grams(self.rouge_calc.tokenize(h, False))
                              for h in summaries[i].split(" . ") if len(h.split(" ")) > 6]
            hyps_tri_grams = {gram: True for l in hyps_tri_grams for gram in l.keys()}

            refs_tri_grams = [self.compute_n_grams(self.rouge_calc.tokenize(r, False))
                              for r in references[i].split(" . ") if len(r.split(" ")) > 3]
            refs_tri_grams = {gram: True for l in refs_tri_grams for gram in l.keys()}

            source_tri_grams = [self.compute_n_grams(self.rouge_calc.tokenize(s, False))
                                for s in sources[i].split(" . ") if len(s.split(" ")) > 3]
            source_tri_grams = {gram: True for l in source_tri_grams for gram in l.keys()}

            novelty_count = 0
            #ref_novelty = sum([1 for gram in refs_tri_grams if gram not in source_tri_grams])
            novel_n_grams = [gram for gram in hyps_tri_grams if gram in refs_tri_grams and gram not in source_tri_grams]
            reward_grams.append(novel_n_grams)
            #print(novel_n_grams)

            if len(refs_tri_grams) == 0: scores.append(0)
            else: scores.append(len(novel_n_grams)/ len(refs_tri_grams))

        if return_grams: return scores, reward_grams
        return scores



class SacreBleu():
    def __init__(self):
        self.bleu_calc = BLEUCalculator()

    def compute_reward(self, samples, sequence, model):
        references = [pair.get_text(pair.full_target_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        summaries = [" ".join([str(token) for token in s]).split(" EOS")[0] for s in sequence]
        scores = []
        for i in range(len(references)):
            scores.append(self.bleu_calc.bleu(summaries[i], references[i]) / 100)
        return scores





class RougeSumeval():
    def __init__(self, rouge_variants, remove_stopwords=False, stem=False):
        self.rouge_variants = rouge_variants
        self.rouge_calc = RougeCalculator(stopwords=remove_stopwords, lang="en", stemming=stem)
        self.availeble_calcs = {"ROUGE-1-F": self.rouge_calc.rouge_1,
                                "ROUGE-2-F": self.rouge_calc.rouge_2,
                                "ROUGE-L-F": self.rouge_calc.rouge_l}

        self.calcs_to_use = []

        for v in self.rouge_variants:
            if v in self.availeble_calcs: self.calcs_to_use.append(self.availeble_calcs[v])
            else: print("Rouge variant not useable", v)

    def compute_reward(self, samples, sequence, model):

        references = [pair.get_text(pair.full_target_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        summaries = [" ".join([str(token) for token in s]).split(" EOS")[0] for s in sequence]

        scores = []
        for i in range(len(references)):
            scores.append(sum([calc(summaries[i], references[i]) for calc in self.calcs_to_use])
                          / len(self.calcs_to_use))

        return scores