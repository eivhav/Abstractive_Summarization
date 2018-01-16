from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def prepareData(pairs):
    input_lang = Lang('headline')
    output_lang = Lang('text_content')
    print("Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


#input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#print(random.choice(pairs))

def load_dailyMail():
    import json
    import glob
    files = list(glob.iglob('/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/tokenized/'+"*.txt"))
    data = dict()
    for f in files:
        d = json.load(open(f))
        for k in d.keys(): data[k] = d[k]
    text_pairs = []
    for el in data.keys():
        text_pairs.append([data[el]['tok_text_content'].lower().replace("\u200f", "")[0:300],
                            data[el]['tok_headline'].lower().replace("\u200f", "")])

    for pair in text_pairs:
        pair[0] = unicodeToAscii(pair[0])
        pair[1] = unicodeToAscii(pair[1])

    return text_pairs



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result


def variablesFromPair(pair, input_lang, output_lang):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


