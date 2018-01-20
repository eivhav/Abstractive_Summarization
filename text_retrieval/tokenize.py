import glob
import json
from multiprocessing import Process
import spacy
import unicodedata
path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/'
out_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/extracted/'
path_toshiba = '/media/havikbot/TOSHIBA EXT/MasterThesis/Data/dailymail/downloads/'
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

def tokenize_data(data, nlp, keys, p_id):
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




if __name__ == '__main__':
    extracted_files = list(glob.iglob(path+"data_json/"+"*.txt"))
    data = dict()
    count = 0
    for file in extracted_files:
        print(file)
        d = json.load(open(file))
        for k in d.keys(): data[k] = d[k]

    nb_processes = 4
    tasks = []
    results = []
    for i in range(nb_processes):
        tasks.append([])
        results.append(dict())

    processes = []
    keys = list(data.keys())
    for i in range(len(keys)): tasks[i % nb_processes].append(keys[i])

    for p in range(nb_processes):
        nlp = spacy.load('en')
        proc = Process(target=tokenize_data, args=(data, nlp, tasks[p], p))
        proc.start()
        proc = processes.append(proc)
    for p in processes: p.join()
