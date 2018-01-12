import json
import glob
files = list(glob.iglob('/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/tokenized/'+"*.txt"))
data = dict()
for f in files:
    d = json.load(open(f))
    for k in d.keys(): data[k] = d[k]

def compute_3_grams(text):
    return  {text[i] + "_"+text[i+1] + "_" + text[i+2]: True for i in range(len(text) -2)}

def compute_novelty(summary, text_content):
    text_3_grams = compute_3_grams(text_content)
    summary_3_grams = compute_3_grams(summary)
    overlapp = [tri_gram for tri_gram in summary_3_grams if tri_gram in text_3_grams]
    return 1 - (len(overlapp) / len(summary_3_grams))

count = 0
total = 0
for d in data:
    count += 1
    novelty = compute_novelty(data[d]['tok_summary_items'], data[d]['tok_text_content'])
    total += novelty
    if novelty > 1.1:
        print(novelty)
        s_items = data[d]['tok_summary_items'].split(".")
        for s in s_items: print("*", s)
        t_items = data[d]['tok_text_content'].split(".")
        for s in t_items: print(s)
        print()

    if count % 2000 == 0: print(count)

print(total/count)


