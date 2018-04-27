from pythonrouge.pythonrouge import Pythonrouge

from sumeval.metrics.rouge import RougeCalculator
import time

class RougePerlVersion():

    def __init__(self, rouge_variants):
        self.rouge_variants = rouge_variants

    def compute_reward(self, samples, sequence, model):
        references = [pair.get_text(pair.full_target_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        summaries = [" ".join(s).split(" EOS")[0] for s in sequence]

        scores = []
        start = time.time()
        for i in range(len(summaries)):
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=[s.replace(" . ", " .\n").split("\n") for s in summaries[i:i+1]],
                                reference=[[s.replace(" . ", " .\n").split("\n")] for s in references[i:i+1]],
                                n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                                recall_only=False, stemming=True, stopwords=False,
                                word_level=True, length_limit=False, length=150,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=False, samples=1000, favor=True, p=0.5)
            score = rouge.calc_score()
            scores.append(sum([float(score[rouge]) for rouge in self.rouge_variants if rouge in score]) \
               / len(self.rouge_variants))
        #print("Computed for ", len(summaries), ": ", time.time() - start, scores)
        return scores


class RougeSumeval():
    def __init__(self, rouge_variants):
        self.rouge_variants = rouge_variants
        self.rouge_calc = RougeCalculator(stopwords=True, lang="en", stemming=True)
        self.availeble_calcs = {"ROUGE-1-F": self.rouge_calc.rouge_1,
                                "ROUGE-2-F": self.rouge_calc.rouge_2,
                                "ROUGE-L-F": self.rouge_calc.rouge_l}

        self.calcs_to_use = []

        for v in self.rouge_variants:
            if v in self.availeble_calcs: self.calcs_to_use.append(self.availeble_calcs[v])
            else: print("Rouge variant not useable", v)

    def compute_reward(self, samples, sequence, model):
        references = [pair.get_text(pair.full_target_tokens, model.vocab).split(" EOS")[0] for pair in samples]
        summaries = [" ".join(s).split(" EOS")[0] for s in sequence]

        scores = []
        for i in range(len(references)):
            scores.append(sum([calc(summaries[i], references[i]) for calc in self.calcs_to_use])
                          / len(self.calcs_to_use))

        return scores