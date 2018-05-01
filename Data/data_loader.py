
import pickle, random
from Data.data import TextPair

class DataLoader:
    def __init__(self, path, limit=45):
        self.path = path
        with open(self.path + "manifest_new.pickle" , 'rb') as f: self.manifest = pickle.load(f)
        with open(self.path + "vocab.pickle" , 'rb') as f: self.vocab = pickle.load(f)
        self.limit = limit

    def sample_training_batch(self, batch_size):
        samples = []
        bucket = random.randint(0, len(self.manifest['training']['buckets'][self.limit])-1)
        for b in range(batch_size):
            file_id = random.choice(self.manifest['training']['buckets'][self.limit][bucket])
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




