from __future__ import unicode_literals, print_function, division


import torch
import pickle
from DeepRL.deepRL_model import PGCModel
#from PGC.model import PGCModel
from DeepRL.data_loader import *

use_cuda = torch.cuda.is_available()

data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/Data/'
path_3 = '/home/havikbot/MasterThesis/Data/NYTcorpus/with_abstract/model_data/'

model_path = '/home/shomea/h/havikbot/MasterThesis/Models/DeepRL/'
data_set_name = 'NYT_40k_summary_v1.pickle'


with open(path_3 +data_set_name, 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.08)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

config = {'embedding_size': 128, 'hidden_size': 256, 'input_length': 300, 'target_length': 50}


pointer_gen_model = PGCModel(config=config, vocab=dataset.vocab, use_cuda=use_cuda,
                             model_path=model_path, model_id='DeepRL')
pointer_gen_model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=50,
                        optimizer=torch.optim.Adam, lr=0.001,
                        tf_ratio=0.5, stop_criterion=None,
                        use_cuda=True, _print=True
                        )
