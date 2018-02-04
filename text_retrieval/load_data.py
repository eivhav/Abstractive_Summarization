


from bs4 import BeautifulSoup
import glob
import json
from multiprocessing import Process
import nltk
import spacy
path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/'
out_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/extracted/'
path_toshiba = '/media/havikbot/TOSHIBA EXT/MasterThesis/Data/dailymail/downloads/'
nyt_path = '/home/havikbot/MasterThesis/Data/NYTcorpus/with_abstract/json/'

files = list(glob.iglob(nyt_path+"*.txt"))
data = dict()
for f in files:
    d = json.load(open(f))
    for k in d.keys(): data[k] = d[k]
vocab = dict()
all_text = [data[k]['tok_text_content'].lower().replace("\u200f", "") for k in data.keys()]

'''
for text in all_text:
    for w in text.replace("  ", " ").split(" "):
        if w not in vocab: vocab[w] = 0
        vocab[w] += 1
'''