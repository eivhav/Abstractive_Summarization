import glob
import json
from multiprocessing import Process
import spacy
import unicodedata
#path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/'
#out_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/extracted/'
#path_toshiba = '/media/havikbot/TOSHIBA EXT/MasterThesis/Data/dailymail/downloads/'
import re




def unicode_to_ascii(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def remove_http_url(text): return ' '.join([w for w in text.split(" ") if '.co' not in w and 'http' not in w])


def split_char_numbers(word_list):
    output = []
    for w in word_list:
        output += [token for token in re.split(r"([A-Za-z'_]+)", w) if token is not '']
    return " ".join(output)

'''
# Todo:
# 1.25billion still a problem.
# Maybe do something about (son _ in _ law, x _ year _ old )
'''





def tokenize_data(path, data, nlp, keys, p_id):
    count = 0
    results = dict()
    for k in keys:
        count += 1
        if count % 1000 == 0:
            with open(path+ "tokenized4/"+"DM_tok_"+str(int(count/1000)) + "000_"+str(p_id)+".txt", "w") as handle:
                handle.write(json.dumps(results))
            results = dict()

        results[k] = data[k].copy()
        for type in ['headline', 'summary_items', 'text_content']:
            if type == 'headline': data[k][type] = [data[k][type]]
            text = ". ".join([s.strip() for s in data[k][type]]).replace("..", ".")
            text = unicode_to_ascii(text).replace("‘", "'").replace("’", "'").replace("  ", " ")
            text = remove_http_url(text)
            text = text.replace("- ", "___").replace("-", "_").replace("___", "-").replace("%", " %")
            doc = nlp(text)
            results[k]['tok_'+type] = split_char_numbers([t.text for t in doc]).replace("   ", " ").replace("  ", " ")

        if count % 500 == 0:
            print(p_id, count, 100*count/len(keys))
            print(data[k])
            for type in ['headline', 'summary_items', 'text_content']:
                print(results[k]['tok_'+type])

    with open(path+ "tokenized/"+"DM_tok_last"+str(p_id)+".txt", "w") as handle:
        handle.write(json.dumps(results))


def clean_nyt(text_elements):
    text_elements = [text.replace("(S)", "").replace("(M)", "").strip() for text in text_elements if len(text) > 15]
    return ". ".join(text_elements).replace("  ", " ")


def tokenize_nyt(path, nlp, articles, p_id):
    count = 0
    results = dict()
    for article, key in articles:
        if 'summary_items' not in article.keys() or 'headline' not in article.keys(): continue
        count += 1
        if count % 1000 == 0:
            with open(path+ "tokenized/"+"NYT_tok_"+str(int(count/1000)) + "000_"+str(p_id)+".txt", "w") as handle:
                handle.write(json.dumps(results))
            results = dict()

        results[key] = article
        for type in ['headline', 'summary_items', 'text_content']:
            if type == 'headline': text = clean_nyt(article[type])
            elif type == 'summary_items': text = clean_nyt(article[type])
            else: text = clean_nyt([article[type]])
            results[key]['tok_'+type] = " ".join([t.text for t in nlp(text)]).replace("' '", "''")

        if count % 1000 == 0:
            print(p_id, count, 100*count/len(keys))
            print(article)
            for type in ['headline', 'summary_items', 'text_content']:
                print(results[key]['tok_'+type])

    with open(path+ "tokenized/"+"NYT_tok_last"+str(p_id)+".txt", "w") as handle:
        handle.write(json.dumps(results))

import time, re

def tokenize_text(nlp, text):
    text = text.replace("(S)", "").replace("(M)", "").replace("‘", "'").replace("’", "'")
    text = remove_http_url(text)
    text = text.replace("   ", " ").replace("  ", " ")
    return " ".join([t.text for t in nlp(text)]).replace("' '", "''")


def fix_tokenized_4(text):
    text = text.replace(". ' .", ". '").replace("' '", "''")
    matchObj = re.search(r'\d\s[_]\w', text)
    while matchObj is not None:
        text = text.replace(matchObj.group(), matchObj.group().replace(" _", "-"))
        matchObj = re.search(r'\d\s[_]\w', text)
    return text.replace("_", " - ").replace("  ", " ")






    '''
    words = text.split()
    result = [words[0]]
    for i in range(1, len(words)):
        if "_" == words[i][0] and words[i-1][-1].isdigit():
            result[i-1] = words[i-1] +
    '''

import unicodedata

def tokenize_data_common(out_path, nlp, articles, p_id, corpus):
    count = 0
    results = dict()
    start = time.time()
    for article, key in articles:
        if 'summary_items' not in article.keys() or 'headline' not in article.keys(): continue
        count += 1
        if count % 1000 == 0:
            with open(out_path+ "tokenized_common/"+corpus+"_tok_"+str(int(count/1000)) + "000_"+str(p_id)+".txt", "w") as handle:
                handle.write(json.dumps(results))
            results = dict()

        results[key] = article
        for type in ['headline', 'summary_items', 'text_content']:
            if isinstance(article[type], list):
                if type == 'summary_items': text = ". ".join(s.strip() for s in article[type] if len(s) > 10)
                elif type == 'text_content': text = " ".join(s.strip() for s in article[type])
                elif type == 'headline': text = article[type][0]
            else:
                text = article[type]
            text = text.replace("(S)", "").replace("(M)", "").replace("‘", "'").replace("’", "'")
            text = unicodedata.normalize("NFKD", text)
            text = remove_http_url(text)
            text = text.replace("   ", " ").replace("  ", " ").replace("-", "_")
            results[key]['tok_' + type] = " ".join([t.text for t in nlp(text)]).replace("' '", "''").replace("_", "-")

        if count % 250 == 0:
            print(p_id, count, 100*count/len(articles), time.time() - start)


    with open(out_path+ "tokenized_common/"+corpus+"_tok_last"+str(p_id)+".txt", "w") as handle:
        handle.write(json.dumps(results))






if __name__ == '__main__':
    for corpus in ['CNN', 'DailyMail']:
        path = "/home/havikbot/MasterThesis/Data/" + corpus + "/"

        extracted_files = list(glob.iglob(path+"extracted/"+"*.txt"))
        data = dict()
        count = 0
        for file in extracted_files:
            d = json.load(open(file))
            for k in d.keys(): data[k] = d[k]

        nb_processes = 2
        tasks = []
        results = []
        for i in range(nb_processes):
            tasks.append([])
            results.append(dict())

        processes = []
        keys = list(data.keys())
        for i in range(len(keys)): tasks[i % nb_processes].append((data[keys[i]], keys[i]))
        data = None

        for p in range(nb_processes):
            nlp = spacy.load('en')
            proc = Process(target=tokenize_data_common, args=(path, nlp, tasks[p], p, corpus))
            proc.start()
            proc = processes.append(proc)
        for p in processes: p.join()



