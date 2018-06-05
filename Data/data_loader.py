
import pickle, random
from Data.data import TextPair
import math

class DataLoader:
    def __init__(self, path, limit=45, filtered=None, s_damp=0.5):
        self.path = path
        with open(self.path + "manifest_new.pickle" , 'rb') as f: self.manifest = pickle.load(f)
        with open(self.path + "vocab.pickle" , 'rb') as f: self.vocab = pickle.load(f)
        self.limit = limit
        self.filtered = filtered
        self.s_damp = s_damp

        new_buckets = []
        for bucket in self.manifest['training']['buckets'][self.limit]:
            new_bucket = [[], []]
            for key in bucket:
                if filtered is not None:
                    nov_degree = self.manifest['training']['samples'][key]['novelty_degrees']
                    if sum(nov_degree) / len(nov_degree) >= filtered:
                            new_bucket[0].append(key)
                            new_bucket[1].append(1 / ((101 - 100*(sum(nov_degree) / len(nov_degree)))**self.s_damp))
                else:
                    new_bucket[0].append(key)
                    new_bucket[1].append(1)
            new_buckets.append(new_bucket)

        if filtered is not None:
            print("NoveltySampling", filtered,
                  sum([len(b) for b in self.manifest['training']['buckets'][self.limit]]), '-->',
                  sum([len(b[0]) for b in new_buckets]) )
        self.manifest['training']['buckets'][self.limit] = new_buckets



    def sample_training_batch(self, batch_size):
        samples = []
        b_int = random.randint(0, len(self.manifest['training']['buckets'][self.limit])-1)
        bucket = self.manifest['training']['buckets'][self.limit][b_int]
        file_ids = random.choices(bucket[0], weights=bucket[1], k=batch_size)
        for file_id in file_ids :
            with open(self.path + "data/"+file_id + ".pickle" , 'rb') as f: sample = pickle.load(f)
            target = [sample['target'][i] for i in range(len(sample['target'])) if
                      sum([len(sample['target'][j]) for j in range(i+1)]) <= self.limit]

            samples.append(TextPair(sample['source'], target, sample['id'], self.vocab,
                                    novelty_vec=sample['n_gram_novelty']))

        return samples


    def load_data(self, data_type, batch_size):
        batch = []
        batches = []
        for file_id in sorted(list(self.manifest[data_type].keys())):
            with open(self.path + "data/" + file_id + ".pickle", 'rb') as f: sample = pickle.load(f)
            batch.append(TextPair(sample['source'], sample['target'], sample['id'], self.vocab,
                                    novelty_vec=sample['n_gram_novelty']))
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []

        return batches




