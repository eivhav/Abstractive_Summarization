
import json, glob, unicodedata
from Data.data import Vocab


class DataSet:
    def __init__(self, name):
        self.name = name
        self.vocab = None
        self.summary_pairs = []
        self.manifest = None

    def add_word(self, word, vocab):
        if word not in vocab.word2count:
            vocab.word2count[word] = 1
            vocab.n_words += 1
        else: vocab.word2count[word] += 1


    def compute_n_gram_novelty(self, source, target, n_gram_n):

        def compute_n_grams(text, n):
            return ["~".join([str(text[i + k]) for k in range(n)]) for i in range(len(text) - n + 1)]

        text_n_grams = {k: True for k in compute_n_grams(source, n_gram_n)}
        summary_n_grams = compute_n_grams(target, n_gram_n)
        nov_vector = [1]*len(target)
        for i in range(2, len(summary_n_grams)):
            if summary_n_grams[i] in text_n_grams: nov_vector[i] = 0

        return nov_vector

    def build_vocab(self, text_pairs, limit):
        vocab = Vocab(limit)
        for pair in text_pairs:
            for w in pair[0].split(" "): self.add_word(w, vocab)
            for w in " ".join(pair[1]).replace("  ", " ").split(" "):
                self.add_word(w, vocab)
        words = [(w, vocab.word2count[w]) for w in vocab.word2count.keys()]
        words = sorted(words, key=lambda tup: tup[1], reverse=True)

        if len(words) > limit: words = words[:limit]
        for w in words:
            index = len(vocab.index2word)
            vocab.index2word[index] = w[0]
            vocab.word2index[w[0]] = index

        vocab.vocab_size = len(vocab.index2word)

        return vocab

    def create_dataset(self, file_paths, source_field, target_field, limit, vocab=None):
        pairs =[]
        for file in file_paths:
            pairs += load_data(file, source_field, target_field)
        if len(file_paths) > 1: pairs = sorted(pairs, key=lambda tup: tup[2])

        print("Data loaded: ", len(pairs))
        if not vocab: self.vocab = self.build_vocab(pairs, limit)
        else: self.vocab = vocab

        print("Vocab built")
        count = 0

        for pair in pairs:
            el = {'source': pair[0].split(" "),
                  'target': [line.split(" ") for line in pair[1]],
                  'time': pair[2], 'id': pair[3]}
            el['n_gram_novelty'] = [self.compute_n_gram_novelty(el['source'][:400], target, n_gram_n=3)
                                    for target in el['target']]
            el['n_gram_novelty_degree'] = [sum(vec)/len(vec) for vec in el['n_gram_novelty']]
            self.summary_pairs.append(el)
            count += 1
            if count % 10000 == 0: print(count)






    def create_manifest(self, nb_buckets):
        manifest = {'training': {'samples': dict(), 'buckets': []}, 'val': dict(), 'test': dict()}
        data = [self.summary_pairs[0:-24858], self.summary_pairs[-24858:-11490], self.summary_pairs[-11490:]]
        dicts = [manifest['training']['samples'], manifest['val'], manifest['test']]
        for i in range(3):
            for pair in data[i]: dicts[i][pair['id']] = {'source_length': len(pair['source']),
                                                        'target_lengths': [len(t) for t in pair['target']],
                                                        'novelty_degrees': [d for d in pair['n_gram_novelty_degree']]}

        sorted_samples = sorted(data[0], key=lambda pair: sum(len(t) for t in pair['target']))
        bucket_size = int(len(sorted_samples) / nb_buckets)
        buckets = [sorted_samples[bucket_size * i:bucket_size * (i + 1)] for i in range(nb_buckets)]
        print(len(buckets))
        for b in range(len(buckets)):
            manifest['training']['buckets'].append([pair['id'] for pair in buckets[b]])

        self.manifest = manifest

    def dump_dataset(self, out_path):
        print('Dumping data')
        import pickle
        with open(out_path + 'manifest.pickle', 'wb') as handle:
            pickle.dump(self.manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_path + 'vocab.pickle', 'wb') as handle:
            pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        count = 0
        for pair in self.summary_pairs:
            count += 1
            with open(out_path + 'data/' + str(pair['id']) + '.pickle', 'wb') as handle:
                pickle.dump(pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if count % 1000 == 0: print(count)








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


        count += 1
        time = '0'
        if 'timestamp_pub' in data[el]: time = data[el]['timestamp_pub']
        text_pairs.append([unicodedata.normalize("NFKD", data[el][source_field].lower()).replace('\xa0', ''),
                           [unicodedata.normalize("NFKD", t.lower()).replace('\xa0', '') for t in data[el][target_field]]
                           , time, el])

        #if count % 1000 == 0: print(count)

    return sorted(text_pairs, key=lambda tup: tup[2])




path_dm_online =  '/home/havikbot/MasterThesis/Data/DailyMail/combined_final/'+"*.txt"
path_cnn_online =  '/home/havikbot/MasterThesis/Data/CNN/combined_final/'+"*.txt"

out_path = '/home/havikbot/MasterThesis/Data/'

dataset = DataSet('DM_CNN_50k_disk')
dataset.create_dataset([path_dm_online, path_cnn_online], 'online_text_content', 'online_summary', 50000)
dataset.create_manifest(100)
dataset.dump_dataset('/home/havikbot/MasterThesis/Data/Model_data/CNN_DM/')

