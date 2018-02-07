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
pointer_gen_model.load_model(file_path='/srv/havikbot/MasterThesis/Models/',
                              file_name='checkpoint_DeepRL_nyt_ep@.pickle')


results = dict()
for i in range(3000, 10000):
    print(i)
    pair = test_pairs[i]
    pred = pointer_gen_model.predict([pair], 75, False, use_cuda)
    pred_beam = pointer_gen_model.predict([pair], 75, 5, use_cuda)
    ref =  pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab).replace(" EOS", "")
    arg_max = " ".join([t[0]['word'] for t in pred if t[0]['word'] != 'EOS' and t[0]['word'] != 'PAD'])
    if len(pred_beam[0][0]) > 15: beam = pred_beam[0][0]
    else: beam = pred_beam[1][0].replace(' EOS', "").replace(" PAD", "")
    results[i] = {'ref': ref, 'greedy': arg_max, 'beam': beam}

import json
with open(samuel_path + "100_preds.json", 'w') as f: f.write(json.dumps(results))




#pointer_gen_model.