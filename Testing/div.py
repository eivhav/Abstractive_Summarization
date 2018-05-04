import glob, json

def load_summaries(file_path):
    files = list(glob.iglob(file_path))
    data = dict()
    for file in files:
        data[file.split("/")[-1].split("_")[0]] = open(file).readlines()
    return data

sums = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/pointer-gen-cov/*.txt')
refrences  = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/reference/*.txt')


count = 0
r_l = 0
for d in sums:
    try:
        hyps, refs = map(list, zip(*[[sums[d][i].strip(), refrences[d][i].strip()] for i in range(min(len(sums[d]), len(refrences[d])))]))
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        r_l += scores['rouge-l']['f']
        count += 1
        if count % 100 == 0:
            print(count)

    except:
        print("Failed")



import glob, json
def load_summaries(file_path):
    files = list(glob.iglob(file_path))
    data = dict()
    for file in files:
        data[file.split("/")[-1].split("_")[0]] = open(file).readlines()
    return data

sumaries = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/pointer-gen-cov/*.txt')
refrences  = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/reference/*.txt')

sums= []
refs = []
for k in sumaries :
    sums.append(" ".join(s.strip() for s in sumaries[k]))
    refs.append(" ".join(s.strip() for s in refrences[k]))



from nltk.stem.porter import *
stemmer = PorterStemmer()
hyps = []
refs = []
for k in sumaries :
    s = " ".join(s.strip() for s in sumaries[k]).strip()
    r = " ".join(s.strip() for s in refrences[k]).strip()
    hyps.append(" ".join([w for w in s.split(" ")]))
    refs.append(" ".join([w for w in r.split(" ")]))

from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)


from pythonrouge.pythonrouge import Pythonrouge
def compute_perl_scores(self, reference, summary):
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=[[summary]],
                        reference=[[[reference]]],
                        n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=False, stemming=False, stopwords=False,
                        word_level=True, length_limit=False, length=150,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=False, samples=1000, favor=True, p=0.5)
    return rouge.calc_score()


from sumeval.metrics.rouge import RougeCalculator
rouge_calc = RougeCalculator(stopwords=False, lang="en", stemming=False)

from nltk.stem.porter import *
stemmer = PorterStemmer()
for i in range(100):
    ref = refs[i].split(" . ")[0]
    hyp = hyps[i].split(" . ")[0]
    print(ref)
    print(hyp)
    perl = compute_perl_scores(None, ref, hyp)

    scores = rouge.get_scores(hyp, ref, avg=True)
    print(perl)
    print(scores)
    '''

    calcs = {"ROUGE-1-F": rouge_calc.rouge_1,
                        "ROUGE-2-F": rouge_calc.rouge_2,
                        "ROUGE-L-F": rouge_calc.rouge_l}
    scores = {k: calcs[k](hyp, ref) for k in calcs}

    for k in calcs:
        print(k, perl[k], scores[k])

    '''
    print()


import glob, json
def load_summaries(file_path):
    files = list(glob.iglob(file_path))
    data = dict()
    for file in files:
        data[file.split("/")[-1].split("_")[0]] = open(file).readlines()
    return data
sumaries = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/pointer-gen-cov/*.txt')
refrences  = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/reference/*.txt')
from nltk.stem.porter import *
stemmer = PorterStemmer()
hyps = []
refs = []
for k in sumaries :
    s = " ".join(s.strip() for s in sumaries[k]).strip()
    r = " ".join(s.strip() for s in refrences[k]).strip()
    hyps.append(" ".join([w for w in s.split(" ")]))
    refs.append(" ".join([w for w in r.split(" ")]))

from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(hyps[0:1], refs[0:1], avg=True)
scores


compute_perl_scores(None, refs[0], hyps[0])