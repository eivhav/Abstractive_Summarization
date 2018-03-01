
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append('/home/'+ 'havikbot/MasterThesis/Code/Abstractive_Summarization/')

import time
from PointerGenerator import data_loader, utils
from PointerGenerator.data_loader import *


class SiameseConvModule(nn.Module):
    def __init__(self, vocab_size, embedding_size, kernels, nb_filters):
        super(SiameseConvModule, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.kernels = kernels
        self.conv_filters = nn.ModuleList(
            [nn.Conv2d(1, nb_filters, (kernel, embedding_size), padding=(int(kernel/2), 0)) for kernel in self.kernels])
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-08)

    def conv_forward(self, input):
        embedded = self.embedding(input)
        convs = [conv_filter(embedded.unsqueeze(1)) for conv_filter in self.conv_filters]
        pooling_layer = nn.MaxPool1d(input.size(-1))
        pools = [pooling_layer(conv.squeeze(-1)) for conv in convs]
        vector = pools[0]
        for pool in pools[1:]: vector = torch.cat((vector, pool), 1)
        return vector.squeeze(-1)

    def forward(self, source_tokens, target_tokens):
        return self.cosine(self.conv_forward(source_tokens), self.conv_forward(target_tokens))


class



class NoveltyModel():
    def __init__(self, config, vocab, use_cuda):
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        self.kernels = config['kernels']
        self.nb_filters = config['nb_filters']

        self.model = SiameseConvModule(self.vocab_size, self.embedding_size, self.kernels, self.nb_filters)
        self.optimizer = None
        self.criterion = None
        self.logger = None
        self.vocab = vocab

        if use_cuda: self.model.cuda()



    def train(self, data, val_data, nb_epochs, batch_size, optimizer, lr, use_cuda, print_evry):

        if self.logger is None:
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=0.0000001)
            self.criterion = torch.nn.L1Loss()
            self.logger = utils.TrainingLogger(nb_epochs, batch_size, len(data), len(val_data))
            print("Optimizers compiled")

        for epoch in range(len(self.logger.log), nb_epochs):
            self.logger.init_epoch(epoch)
            batches = utils.sort_and_shuffle_data(data, nb_buckets=100, batch_size=batch_size, rnd=False)
            for b in range(len(batches)):
                loss, _time = self.train_batch(samples=batches[b], use_cuda=use_cuda)
                self.logger.add_iteration(b + 1, loss, _time)

            for b in range(int(len(val_data) / batch_size)):
                loss, _time = self.train_batch(val_data[b * batch_size:(b + 1) * batch_size], use_cuda, backprop=False)
                self.logger.add_val_iteration(b + 1, loss, _time)



    def train_batch(self, samples, use_cuda, backprop=True):
        start = time.time()
        input_variable, full_input_variable, target_variable, full_target_variable, decoder_input = \
                        utils.get_batch_variables(samples, 400, 50, use_cuda, self.vocab.word2index['SOS'])

        novelty_score = Variable(torch.FloatTensor(self.compute_novelty_n_grams(samples, 3)))
        if use_cuda: novelty_score = novelty_score.cuda()

        loss = self.criterion(self.model(full_input_variable, full_target_variable), novelty_score)
        if backprop:
            loss.backward()
            self.optimizer.step()
        return loss.data[0], time.time() - start



    def compute_novelty_n_grams(self, pairs, n_gram):

        def compute_n_grams(tokens, n):
            return {"~".join([str(tokens[i + k]) for k in range(n)]): True for i in range(len(tokens) - n + 1)}

        novelties = []
        for pair in pairs:
            text_n_grams = compute_n_grams(pair.full_source_tokens, n_gram)
            summary_n_grams = compute_n_grams(pair.full_target_tokens, n_gram)
            overlapp = [tri_gram for tri_gram in summary_n_grams if tri_gram in text_n_grams]

            if len(summary_n_grams) == 0:
                novelties.append(1)
            else:
                novelty = 1 - (len(overlapp) / len(summary_n_grams))
                novelties.append(novelty)

        return novelties


import pickle
use_cuda = torch.cuda.is_available()

data_path = 'havikbot/MasterThesis/Data/'
model_path = 'havikbot/MasterThesis/Models/'
dataset_dm_cnn = 'DM_cnn_50k.pickle'

with open('/home/' + data_path +dataset_dm_cnn , 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.9)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.9):]

config = {'vocab_size': dataset.vocab.vocab_incl_size+1000,
          'embedding_size': 200, 'kernels': [3, 4, 5, 7, 9], 'nb_filters':500}
model = NoveltyModel(config, vocab= dataset.vocab, use_cuda=use_cuda)

model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=50,
                        optimizer=torch.optim.Adam, lr=0.001,
                        use_cuda=True, print_evry=200
                        )






