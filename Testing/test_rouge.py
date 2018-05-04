


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
    hyps.append(" ".join(s.strip() for s in sumaries[k]).strip())
    refs.append(" ".join(s.strip() for s in refrences[k]).strip())


stem = False
remove_stop = True

from pythonrouge.pythonrouge import Pythonrouge
def compute_perl_scores(reference, summary, stem, remove_stop):
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=[[summary]],
                        reference=[[[reference]]],
                        n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=False, stemming=stem, stopwords=remove_stop,
                        word_level=True, length_limit=False, length=150,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=False, samples=1000, favor=True, p=0.5)
    return rouge.calc_score()


from sumeval.metrics.rouge import RougeCalculator
rouge_calc = RougeCalculator(stopwords=remove_stop, lang="en", stemming=stem)

from rouge import Rouge
rouge = Rouge(stem=stem, remove_stop=remove_stop)

from nltk.stem.porter import *
stemmer = PorterStemmer()
for i in range(100):
    ref = refs[i]
    hyp = hyps[i]
    #perl = compute_perl_scores(ref, hyp, stem, remove_stop)
    perl = compute_perl_scores(ref.replace(" . ", " .\n"), hyp.replace(" . ", " .\n"), stem, remove_stop)

    s = rouge.get_scores([hyp.replace(" . ", " ,  ")], [ref.replace(" . ", " ,  ")], avg=True)
    t_scores = {"ROUGE-1-F": s['rouge-1']['f'], "ROUGE-2-F": s['rouge-2']['f'], "ROUGE-L-F": s['rouge-l']['f']}
    t_recall = {"ROUGE-1-R": s['rouge-1']['r'], "ROUGE-2-R": s['rouge-2']['r'], "ROUGE-L-R": s['rouge-l']['r']}


    rouge_calcs = {"ROUGE-1-F": rouge_calc.rouge_1,
            "ROUGE-2-F": rouge_calc.rouge_2,
            "ROUGE-L-F": rouge_calc.rouge_l}
    scores = {k: rouge_calcs[k](hyp, ref) for k in rouge_calcs}

    for k in rouge_calcs:
        print(k, round(perl[k], 3), round(t_scores[k], 3), round(scores[k],3))
    for k in t_recall:
        print(k, round(perl[k], 3), round(t_recall[k], 3))



    print()
