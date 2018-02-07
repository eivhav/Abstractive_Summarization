from __future__ import unicode_literals, print_function, division

import pickle

import torch

from PointerGenerator.model import *
from PointerGenerator.data_loader import *

use_cuda = torch.cuda.is_available()

norm_path = '/home/havikbot/MasterThesis/Data/'
samuel_path = '/srv/havikbot/MasterThesis/Data/'
model_path = '/srv/havikbot/MasterThesis/Models/'
data_set_name_dm = 'DM_25k_summary_v2.pickle'
data_set_name_nyt = 'NYT_40k_summary_v3.pickle'


with open(samuel_path  +data_set_name_nyt, 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.8)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

# 'TemporalAttn' or CoverageAttn

config = {'model_type': 'CoverageAttn',
          'embedding_size': 200, 'hidden_size': 400,
          'input_length': 300, 'target_length': 50,
          'model_path': model_path, 'model_id': 'CombinedTest' }


pointer_gen_model = PGModel(config=config, vocab=dataset.vocab, use_cuda=use_cuda)

pointer_gen_model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=50,
                        optimizer=torch.optim.Adam, lr=0.0001,
                        tf_ratio=0.6, stop_criterion=None,
                        use_cuda=True, print_evry=50
                        )

