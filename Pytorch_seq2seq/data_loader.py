
import json
import glob

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import operator
PAD_token = 0
SOS_token = 1
EOS_token = 2


class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4  # Count SOS and EOS



class TextPair:
    def __init__(self, source_text, target_text, limit, vocab):
        self.source_text = source_text
        self.target_text = target_text
        self.source_idx_tokens = []
        self.target_idx_tokens = []
        self.unknown_tokens = dict()

        self.add_sentence(source_text, self.source_idx_tokens, vocab)
        self.add_sentence(target_text, self.target_idx_tokens, vocab)

    def add_sentence(self, sentence, destination, vocab):
        for w in sentence.split(" "):
            try:
                if w in vocab.word2index: destination.append(vocab.word2index[w])
                else:
                    destination.append(len(vocab.index2word) + len(self.unknown_tokens))
                    self.unknown_tokens[w] = 1

            except:
                #print(w)
                continue
        destination.append(EOS_token)


class DataSet:
    def __init__(self, name):
        self.name = name
        self.vocab = None
        self.summary_pairs = []

    def add_word(self, word, vocab):
        if word not in vocab.word2count:
            vocab.word2count[word] = 1
            vocab.n_words += 1
        else: vocab.word2count[word] += 1

    def create_dataset(self, file_path, source_field, target_field, limit):
        pairs = load_data(file_path, source_field, target_field)
        print("Data loaded: ", len(pairs))
        self.vocab = self.build_vocab(pairs, limit)
        print("Vocab built")
        count = 0
        for pair in pairs:
            text_pair = TextPair(pair[0], pair[1], limit, self.vocab)
            self.summary_pairs.append(text_pair)
            count += 1
            if count % 10000 == 0: print(count)

    def build_vocab(self, text_pairs, limit):
        vocab = Vocab()
        for pair in text_pairs:
            for w in pair[0].split(" "): self.add_word(w, vocab)
            for w in pair[1].split(" "): self.add_word(w, vocab)
        words = [(w, vocab.word2count[w]) for w in vocab.word2count.keys()]
        sorted(words, key=lambda tup: tup[1], reverse=True)

        if len(words) > limit: words = words[:limit]
        for w in words:
            index = len(vocab.index2word)
            vocab.index2word[index] = w[0]
            vocab.word2index = w[0]

        return vocab







def load_data(file_path, source_field, target_field):
    files = list(glob.iglob(file_path))
    data = dict()
    for f in files:
        d = json.load(open(f))
        for k in d.keys(): data[k] = d[k]
    text_pairs = []
    count = 0

    def unicodeToAscii(s):
        return s
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    for el in data.keys():
        count += 1
        text_pairs.append([unicodeToAscii(data[el][source_field].lower().replace("\u200f", "")),
                            unicodeToAscii(data[el][target_field].lower().replace("\u200f", ""))])
        #if count % 1000 == 0: print(count)
    return text_pairs



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = indexes
    return result


def variablesFromPair(pair, input_lang, output_lang):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/tokenized/'+"*.txt"
dataset = DataSet('DailyMail')
dataset.create_dataset(path, 'tok_text_content', 'tok_headline', 50000)
