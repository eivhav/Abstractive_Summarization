from __future__ import unicode_literals, print_function, division

import pickle

import torch
import sys, os

samuel = '/srv/'
x99 = '/home/'
current =samuel

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
data_set_name_nyt = 'NYT_40k_summary_v3.pickle'
data_set_nyt_filtered = 'NYT_40k_filtered_v1.pickle'

with open(current + data_path +data_set_nyt_filtered  , 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.9)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.9):]

# 'TemporalAttn' or CoverageAttn

config = {'model_type': 'CoverageAttn',
          'embedding_size': 200, 'hidden_size': 400,
          'input_length': 400, 'target_length': 50,
          'model_path': model_path, 'model_id': 'CombinedTest' }


pointer_gen_model = PGModel(config=config, vocab=dataset.vocab, use_cuda=use_cuda)
pointer_gen_model.load_model(file_path='/srv/havikbot/MasterThesis/Models/',
                              file_name='checkpoint_PGC_NYTfiltered_9_feb_ep@9_loss@2859.406.pickle')


results = dict()
for i in range(1010, 1015 ):
    print(i)
    pair = test_pairs[i]
    pred = pointer_gen_model.predict([pair], 75, False, use_cuda)
    pred_beam = pointer_gen_model.predict([pair], 75, 5, use_cuda)
    ref =  pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab).replace(" EOS", "")
    #print(pred)
    arg_max = " ".join([t[0]['word']+"("+str(round(t[0]['p_gen'], 2))+")" for t in pred if t[0]['word'] != 'EOS' and t[0]['word'] != 'PAD'])
    if len(pred_beam[0][0]) > 15: beam = pred_beam[0][0]
    else: beam = pred_beam[1][0].replace(' EOS', "").replace(" PAD", "")
    results[i] = {'ref': ref, 'greedy': arg_max, 'beam': beam}
    print('ref:', ref)
    print('greedy:',  arg_max)
    print('beam:', beam)

import json
with open(current + data_path + "100_preds.json", 'w') as f: f.write(json.dumps(results))




#pointer_gen_model.