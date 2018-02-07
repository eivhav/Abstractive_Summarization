from __future__ import unicode_literals, print_function, division
import json
import glob


from io import open
import unicodedata
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
    files = list(glob.iglob(file_path))
    data = dict()
    for f in files:
        d = json.load(open(f))
        for k in d.keys(): data[k] = d[k]
    text_pairs = []
    count = 0

    for el in data.keys():
        count += 1
        text_pairs.append([unicodedata.normalize("NFKD", data[el][source_field].lower()),
                           unicodedata.normalize("NFKD", data[el][target_field].lower())])
        #if count % 1000 == 0: print(count)
    return text_pairs


path = '/srv/havikbot/MasterThesis/Data/filtered/'+"*.txt"
out_path = '/srv/havikbot/MasterThesis/Data/'

dataset = DataSet('NYT_filtered')

def create_and_save_dataset(dataset):
    dataset = DataSet('NYtimes_filtered_40k')
    dataset.create_dataset(path, 'tok_text_content', 'tok_summary_items', 40000)

    import pickle
    with open(out_path + 'NYT_40k_filtered_v1.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)



