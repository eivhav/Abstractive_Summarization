import json
import glob
import unicodedata
files = list(glob.iglob('/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/tokenized4/'+"*.txt"))
data = dict()
for f in files:
    d = json.load(open(f))
    for k in d.keys(): data[k] = d[k]

def compute_n_grams(text, n):
    return {"~".join([text[i+k] for k in range(n)]): True for i in range(len(text) -n+1)}


def process_and_split_text(text):
    return [t for t in text.lower().replace(" '", "").replace(", ", "").replace(" _", " ").split(" ") if t != ""]


def unicode_norm(s): return unicodedata.normalize("NFKD", s)


def compute_novelty(summary, text_content, n_gram_n):
    text_n_grams = compute_n_grams(process_and_split_text(unicode_norm(text_content)), n_gram_n)
    summary_n_grams = compute_n_grams(process_and_split_text(unicode_norm(summary)), n_gram_n)
    overlapp = [tri_gram for tri_gram in summary_n_grams if tri_gram in text_n_grams]



    #print(summary)
    #print([tri_gram for tri_gram in summary_n_grams if tri_gram not in overlapp])
    #print(overlapp)
    if len(summary_n_grams) == 0: return -1
    novelty = 1 - (len(overlapp) / len(summary_n_grams))
    if True:
        '''
        print(text_content)
        print(summary)
        print([tri_gram for tri_gram in summary_n_grams if tri_gram not in overlapp])
        print(overlapp)
        print()
        '''
        nb_insertions = 0
        summary_tokens = process_and_split_text(summary)
        n_gram_found = False
        for pos in range(len(summary_tokens) -n_gram_n+1):
            current_n_gram = "~".join(summary_tokens[pos:pos+n_gram_n])
            if current_n_gram not in overlapp:
                nb_insertions += 1
                if n_gram_found:
                    pos += 2
                    n_gram_found = False
            else:
                n_gram_found = True

        novelty = nb_insertions / (len(summary_tokens) -n_gram_n+1)
        if novelty < 0.4:
            print(text_content)
            print(summary, novelty)
            print()

    return novelty

count = 0
total = 0
nov_dist = [0] * 12
for d in data:

    novelty = [compute_novelty(s, data[d]['tok_text_content'], n_gram_n= 3) for s in data[d]['tok_summary_items'].split(" . ")]
    for n in novelty:
        if n == -1: continue
        total += n
        nov_dist[int(n*10)] += 1

    count += len(novelty)

    if count % 2000 == 0: print(count, total/count)
    #if count > 100:break

print(total/count)


