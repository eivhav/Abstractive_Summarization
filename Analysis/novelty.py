import json
import glob
files = list(glob.iglob('/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/tokenized/'+"*.txt"))
data = dict()
for f in files:
    d = json.load(open(f))
    for k in d.keys(): data[k] = d[k]

def compute_n_grams(text, n):
    return {"_".join([text[i+k] for k in range(n)]): True for i in range(len(text) -n)}

def compute_novelty(summary, text_content, n_gram_n):
    text_n_grams = compute_n_grams(text_content.split(" "), n_gram_n)
    summary_n_grams = compute_n_grams(summary.split(" "), n_gram_n)
    overlapp = [tri_gram for tri_gram in summary_n_grams if tri_gram in text_n_grams]
    if len(summary_n_grams) == 0: return -1
    return 1 - (len(overlapp) / len(summary_n_grams))

count = 0
total = 0
nov_dist = [0] * 12
for d in data:
    count += 1
    novelty = compute_novelty(data[d]['tok_summary_items'], data[d]['tok_text_content'], n_gram_n= 2)
    if novelty == -1: continue
    total += novelty
    nov_dist[int(novelty*10)] += 1
    if novelty > 0.8 and True:
        print(novelty)
        s_items = data[d]['tok_summary_items'].split(".")
        for s in s_items: print("*", s)
        t_items = data[d]['tok_text_content'].split(".")
        for s in t_items: print(s)
        print()

    if count % 2000 == 0: print(count, total/count)

print(total/count)


