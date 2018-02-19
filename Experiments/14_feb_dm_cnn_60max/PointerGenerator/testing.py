from __future__ import unicode_literals, print_function, division

import pickle
import spacy

import torch
import sys, os

samuel = '/srv/'
x99 = '/home/'
current =samuel
exp_path = 'Experiments/14_feb_dm_cnn_60max/'

sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/' + exp_path)
sys.path.pop(-2)
sys.path.pop(-2)
sys.path.pop(1)

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
dataset_dm_cnn = 'DM_cnn_50k.pickle'


with open(current + data_path +dataset_dm_cnn  , 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.9)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.9):]

# 'TemporalAttn' or CoverageAttn

config = {'model_type': 'TemporalAttn',
          'embedding_size': 100, 'hidden_size': 400,
          'input_length': 400, 'target_length': 60,
          'model_path': model_path, 'model_id': 'CombinedTest' }


pointer_gen_model = PGModel(config=config, vocab=dataset.vocab, use_cuda=use_cuda)
pointer_gen_model.load_model(file_path='/srv/havikbot/MasterThesis/Models/',
                              file_name='checkpoint_DM_CNN_50k_temporal_14_feb_60max_tf06_ep@13_loss@2536.066.pickle')




def remove_http_url(text): return ' '.join([w for w in text.split(" ") if '.co' not in w and 'http' not in w])


def tokenize_text(nlp, text):
    text = text.replace("(S)", "").replace("(M)", "").replace("‘", "'").replace("’", "'")
    text = remove_http_url(text)
    text = text.replace("   ", " ").replace("  ", " ")
    return " ".join([t.text for t in nlp(text)]).replace("' '", "''")





def predict_and_print(pair, model, limit):
    pred = model.predict([pair], limit, False, use_cuda)
    pred_beam = model.predict([pair], limit, 5, use_cuda)
    ref = pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab).replace(" EOS", "")
    arg_max = " ".join([t[0]['word']  for t in pred if t[0]['word'] != 'EOS' and t[0]['word'] != 'PAD'])
    if len(pred_beam[0][0]) > 15:
        beam = pred_beam[0][0]
    else:
        beam = pred_beam[1][0].replace(' EOS', "").replace(" PAD", "")
    results = {'ref': ref, 'greedy': arg_max, 'beam': beam}
    print('ref:', ref)
    print('greedy:', arg_max)
    print('beam:', beam)
    return results

def test_on_new_article(path, file_name, text, model, vocab):
    nlp = spacy.load('en')
    if text is None:
        text = " ".join(open(path + file_name, 'r').readlines())
    text = tokenize_text(nlp, text)
    text_pair = TextPair(text, '', 1000, vocab)
    result = predict_and_print(text_pair, model, limit=75)

def predict_from_data(test_pairs, _range=(1010, 1015), model=None):
    results = dict()
    for i in range(_range[0], _range[1]):
        print(i)
        pair = test_pairs[i]
        results[i] = predict_and_print(pair, model, 75)
    return results

def save_predictions(result_dict, path, name):
    import json
    with open(path + name+".json", 'w') as f: f.write(json.dumps(result_dict))





#pointer_gen_model.



#pointer_gen_model.