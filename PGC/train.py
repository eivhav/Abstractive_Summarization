from __future__ import unicode_literals, print_function, division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle

#from PGC.model import EncoderRNN, AttnDecoderRNN
import PGC.utils as utils
from PGC.model import PGCModel

use_cuda = torch.cuda.is_available()
multi_gpu = False
from Pytorch_seq2seq.data_loader import *
data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/'

with open(path_2 +'DM_25k_summary.pickle', 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.8)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

pointer_gen_model = PGCModel( config=None, vocab=dataset.vocab, use_cuda=True)
pointer_gen_model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=20, batch_size=32,
                        optimizer=torch.optim.Adagrad, lr=0.015,
                        tf_ratio=0.5, stop_criterion=None,
                        use_cuda=True
                        )
