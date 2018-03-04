from __future__ import unicode_literals, print_function, division
import json
import glob


from io import open
import unicodedata
import random
import operator
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Need to remove \xa0 string = string.replace(u'\xa0', u' ')
# new_str = unicodedata.normalize("NFKD", unicode_str)


class Vocab:
    def __init__(self, limit):
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.word2index = {self.index2word[k]: k for k in self.index2word}
        self.word2count = {}
        self.n_words = 4  # Count SOS and EOS
        self.vocab_size = 0
        self.vocab_incl_size = len(self.index2word) + limit



class TextPair:
    def __init__(self, source, target, limit, vocab):
        self.full_source_tokens = []
        self.masked_source_tokens = []
        self.full_target_tokens = []
        self.masked_target_tokens = []
        self.unknown_tokens = dict()

        self.add_sentence(source, (self.full_source_tokens, self.masked_source_tokens), vocab)
        self.add_sentence(target, (self.full_target_tokens, self.masked_target_tokens), vocab)

    def get_text(self, tokens, vocab):
        words = []
        for t in tokens:
            if t in vocab.index2word: words.append(vocab.index2word[t])
            else: [words.append(w) for w in self.unknown_tokens if self.unknown_tokens[w] == t]
        return " ".join(words)


    def add_sentence(self, sentence, destination, vocab):
        for w in sentence.split(" "):
            try:
                if w in vocab.word2index:
                    destination[0].append(vocab.word2index[w])
                    destination[1].append(vocab.word2index[w])
                else:
                    if w not in self.unknown_tokens:
                        self.unknown_tokens[w] = len(vocab.index2word) + len(self.unknown_tokens)
                    destination[0].append(self.unknown_tokens[w])
                    destination[1].append(UNK_token)
            except:
                print("FAILED FOR WORD", w)
                continue
        destination[0].append(EOS_token)
        destination[1].append(EOS_token)


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

    def create_dataset(self, file_paths, source_field, target_field, limit):
        pairs =[]
        for file in file_paths:
            pairs += load_data(file, source_field, target_field)
        if len(file_paths) > 1: pairs = sorted(pairs, key=lambda tup: tup[2])
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
        vocab = Vocab(limit)
        for pair in text_pairs:
            for w in pair[0].split(" "): self.add_word(w, vocab)
            for w in pair[1].split(" "): self.add_word(w, vocab)
        words = [(w, vocab.word2count[w]) for w in vocab.word2count.keys()]
        words = sorted(words, key=lambda tup: tup[1], reverse=True)

        if len(words) > limit: words = words[:limit]
        for w in words:
            index = len(vocab.index2word)
            vocab.index2word[index] = w[0]
            vocab.word2index[w[0]] = index

        vocab.vocab_size = len(vocab.index2word)

        return vocab





def load_data(file_path, source_field, target_field):
    print("Loading file_path", file_path)
    files = list(glob.iglob(file_path))
    data = dict()
    for f in files:
        d = json.load(open(f))
        for k in d.keys(): data[k] = d[k]
    text_pairs = []
    count = 0

    for el in data.keys():
        try:
            count += 1
            time = '0'
            if 'timestamp_pub' in data[el]: time = data[el]['timestamp_pub']
            text_pairs.append([unicodedata.normalize("NFKD", data[el][source_field].lower()).replace('\xa0', ''),
                           unicodedata.normalize("NFKD", data[el][target_field].lower()).replace('\xa0', '')
                           , time])
        except:
            print(data[el])
        #if count % 1000 == 0: print(count)

    return sorted(text_pairs, key=lambda tup: tup[2])


path_dm = '/home/havikbot/MasterThesis/Data/DailyMail/tokenized_c/'+"*.txt"
path_cnn = '/home/havikbot/MasterThesis/Data/CNN/tokenized_c/'+"*.txt"

path_dm_online =  '/home/havikbot/MasterThesis/Data/DailyMail/combined_2/'+"*.txt"
path_cnn_online =  '/home/havikbot/MasterThesis/Data/CNN/combined_2/'+"*.txt"

out_path = '/home/havikbot/MasterThesis/Data/'

dataset = DataSet('DM_CNN_50k_online')

def create_and_save_dataset(dataset):
    dataset = DataSet('DM_CNN_50k_org')
    dataset.create_dataset([path_dm_online, path_cnn_online], 'online_text_content', 'online_summary', 50000)

    import pickle
    with open(out_path + 'DM_cnn_50k_online.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)



