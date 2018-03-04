from __future__ import unicode_literals, print_function, division

import pickle

import torch
import sys, os

samuel = '/srv/'
x99 = '/home/'
current = x99

sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/')

from PointerGenerator.model import *
from PointerGenerator.data_loader import *
from PointerGenerator.rl_model import *
#from model import *
#from data_loader import *

use_cuda = torch.cuda.is_available()

data_path = 'havikbot/MasterThesis/Data/'
model_path = 'havikbot/MasterThesis/Models/'

data_set_name_dm = 'DM_25k_summary_v2.pickle'
data_dm_v2 = 'DM_50k.pickle'
data_set_name_nyt = 'NYT_40k_summary_v3.pickle'
data_set_nyt_filtered = 'NYT_40k_filtered_v1.pickle'
dataset_dm_cnn = 'DM_cnn_50k.pickle'
dataset_dm_cnn_v2 = 'DM_cnn_50k_online.pickle'

with open(current + data_path +dataset_dm_cnn_v2 , 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.9)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.9):]

# 'TemporalAttn' or CoverageAttn

config = {'model_type': 'CoverageAttn',
          'embedding_size': 128, 'hidden_size': 256,
          'input_length': 400, 'target_length': 100,
          'model_path': current+ model_path, 'model_id': 'DM_CNN_50k_TemporalAttn_02_mars' }


pointer_gen_model = PGmodel_reinforcement(config=config, vocab=dataset.vocab, use_cuda=use_cuda)

'''
pointer_gen_model.load_model(file_path=current + 'havikbot/MasterThesis/Models/',
                              file_name='checkpoint_DM_CNN_50k_CoverageAttn_26_feb_ep@8_loss@4049.408.pickle')
pointer_gen_model.config['target_length'] = 70


'''

pointer_gen_model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=20,
                        optimizer=torch.optim.Adam, lr=0.001,
                        tf_ratio=0.75, stop_criterion=None,
                        use_cuda=True, print_evry=20
                        )

