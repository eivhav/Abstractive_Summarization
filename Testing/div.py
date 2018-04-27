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

sums = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/pointer-gen-cov/*.txt')
refrences  = load_summaries('/home/havikbot/MasterThesis/Data/Abi See- PointerGen test output/reference/*.txt')

summary = []
reference = []
for k in sums:
    summary.append(["\n".join(s.strip() for s in sums[k])])
    reference.append([["\n".join(s.strip() for s in refrences[k])]])



from pythonrouge.pythonrouge import Pythonrouge
import time
start = time.time()
rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary[:1600], reference=reference[:1600],
                    n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                    recall_only=False, stemming=True, stopwords=False,
                    word_level=True, length_limit=False, length=150,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=False, samples=1000, favor=True, p=0.5)
print(time.time() - start)
score = rouge.calc_score()
print(time.time() - start)