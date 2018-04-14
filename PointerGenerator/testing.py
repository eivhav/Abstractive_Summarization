from __future__ import unicode_literals, print_function, division

import pickle

import torch
import sys, os
import spacy

samuel = '/srv/'
x99 = '/home/'
current =x99

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

config = {'model_type': 'TemporalAttn',
          'embedding_size': 128, 'hidden_size': 256,
          'input_length': 400, 'target_length': 60,
          'model_path': current+ model_path, 'model_id': 'DM_CNN_50k_Temporal_novelty' }


pointer_gen_model = PGmodel_reinforcement(config=config, vocab=dataset.vocab, use_cuda=use_cuda)
#pointer_gen_model.load_model(file_path=current + 'havikbot/MasterThesis/Models/',
                              #file_name='checkpoint_DM_CNN_50k_CoverageAttn_26_feb_ep@8_loss@4049.408.pickle')

'''
pointer_gen_model.train_rl(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=32,
                        optimizer=torch.optim.Adam, lr=0.00005,
                        tf_ratio=0.75, stop_criterion=None,
                        use_cuda=True, print_evry=200
                        )


'''


def remove_http_url(text): return ' '.join([w for w in text.split(" ") if '.co' not in w and 'http' not in w])


def tokenize_text(nlp, text):
    text = text.replace("(S)", "").replace("(M)", "").replace("‘", "'").replace("’", "'")
    text = remove_http_url(text)
    text = text.replace("   ", " ").replace("  ", " ")
    return " ".join([t.text for t in nlp(text)]).replace("' '", "''")






def predict_and_print(pair, model, limit):
    #pred = model.predict([pair], limit, False, use_cuda)
    pred_beam = model.predict([pair], limit, 5, use_cuda)
    ref = pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab).replace(" EOS", "")
    #arg_max = " ".join([t[0]['word']  for t in pred if t[0]['word'] != 'EOS' and t[0]['word'] != 'PAD'])
    arg_max = ""
    if len(pred_beam[0][0]) > 15:
        beam = pred_beam[0][0]
    else:
        beam = pred_beam[1][0].replace(' EOS', "").replace(" PAD", "")
    results = {'ref': ref, 'greedy': arg_max, 'beam': beam}
    return results

def test_on_new_article(path, file_name, text, model, vocab):
    nlp = spacy.load('en')
    if text is None:
        text = " ".join(open(path + file_name, 'r').readlines())
    text = tokenize_text(nlp, text)
    text_pair = TextPair(text, '', 1000, vocab)
    result = predict_and_print(text_pair, model, limit=75)

def predict_from_data(test_pairs, _range=(1010, 1100), model=None):
    results = dict()
    count = 0
    for i in range(_range[0], _range[1]):
        count += 1
        #if count % 25 == 0: print(count)
        pair = test_pairs[i]
        results[i] = predict_and_print(pair, model, 75)
    return results

def save_predictions(result_dict, path, name):
    import json
    with open(path + name+".json", 'w') as f: f.write(json.dumps(result_dict))


def predict_greedy(test_pairs, _range=(0, 100), model=None, batch_size=50):
    pairs = test_pairs[_range[0]:_range[1]]
    results = dict()
    for i in range(int(len(pairs) / batch_size)):
        preds = model.predict(pairs[i*batch_size:(i+1)*batch_size], 75, False, use_cuda)
        for p in range(batch_size):
            pair = pairs[(i * batch_size) + p]
            ref = pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab).replace(" EOS", "")
            seq = [t[p] for t in preds]
            arg_max = " ".join([s['word'] for s in seq if s['word'] != 'EOS' and s['word'] != 'PAD'])
            full_text = pair.get_text(pair.full_source_tokens, pointer_gen_model.vocab).replace(" EOS", "")
            #for s in seq: print(s)
            results[(i*batch_size) + p] = {'ref': ref, 'greedy': arg_max, 'beam': '', 'text': full_text}

    return results


from Analysis import novelty as novelty
from nltk.stem.porter import *
stemmer = PorterStemmer()

from PyRouge.pyrouge import Rouge

def score_model(test_pairs, model, model_id, nb_examples, output_type):
    scores = [0, 0, 0, 0]
    rouge_calc = RougeCalculator(stopwords=True, lang="en")
    pyRouge = Rouge()
    if output_type == 'greedy':
        results = predict_greedy(test_pairs, _range=(0, nb_examples), model= model)
    else:
        results = predict_from_data(test_pairs, _range=(0, nb_examples), model= model)
    summaries = []
    novelty_dist = []
    for d in range(11): novelty_dist.append([])
    for k in results:
        el = results[k]
        ref = " ".join([t for t in el['ref'].split('EOS')[0].split(" ")])
        summary = " ".join([t for t in el[output_type].split('EOS')[0].split(" ")])

        scores[0] += rouge_calc.rouge_1(summary , ref)
        scores[1] += rouge_calc.rouge_2(summary , ref)
        rouge_l = rouge_calc.rouge_l(summary , ref)
        ''''''
        n = novelty.compute_novelty(ref, el['text'], 3)
        novelty_dist[int(n*10)].append(rouge_l)

        '''
        print(round(rouge_calc.rouge_2(summary , ref), 3), round(rouge_l, 3), len(summary.split(" ")), summary)
        print(ref)
        print(rouge_calc.rouge_2(summary , ref), rouge_l)
        print(summary)
        print(ref)
        print()
        '''
        scores[2] += rouge_l
        if rouge_l  < 0.20 or True: summaries.append((rouge_l, el[output_type].split('EOS')[0]))

    for i in range(10):
        print((i+1) *10, sum(novelty_dist[i]) / len(novelty_dist[i]), len(novelty_dist[i]))
    '''
    print(model_id.split("@"), round(scores[0]/len(results), 3),
          round(scores[1]/len(results), 3), round(scores[2]/len(results), 3), round(scores[3]/len(results), 3))
    #for s in summaries: print(s)
    #print()
    '''


config = {'model_type': 'TemporalAttn',
          'embedding_size': 128, 'hidden_size': 256,
          'input_length': 400, 'target_length': 60,
          'model_path': current+ model_path, 'model_id': 'DM_CNN_50k_Temporal_novelty' }
import glob

for file_name in sorted(list(glob.iglob('/home/havikbot/MasterThesis/Models/temporal/best/' +"*.pickle"))):
    file_path = file_name.split("check")[0]
    model_id = file_name.split("/")[-1]

    pointer_gen_model = PGmodel_reinforcement(config=config, vocab=dataset.vocab, use_cuda=use_cuda)
    pointer_gen_model.load_model(file_path=file_path, file_name=model_id)
    score_model(test_pairs, model=pointer_gen_model, model_id =model_id, nb_examples=10000, output_type='greedy')




'''
pointer_gen_model.train_rl(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=32,
                        optimizer=torch.optim.Adam, lr=0.00005,
                        tf_ratio=0.75, stop_criterion=None,
                        use_cuda=True, print_evry=200
                        )


'''

def load_summaries(path):
    summaries = dict()
    for file_name in sorted(list(glob.iglob(path+"*.txt"))):
        id = file_name.split("/")[-1].split("_")[0]
        text = " ".join([line.strip() for line in open(file_name).readlines()])
        summaries[id] = text.replace("  ", " ")
    return summaries

def test_summaries(refs, summaries):
    rouge_calc = RougeCalculator(stopwords=False, stemming=False, lang="en")
    scores = [0, 0, 0]
    c = 0
    for k in refs:
        if k in summaries:
            scores[0] += rouge_calc.rouge_1(summaries[k], refs[k])
            scores[1] += rouge_calc.rouge_2(summaries[k], refs[k])
            scores[2] += rouge_calc.rouge_l(summaries[k], refs[k])
        c += 1

    return [scores[0] / c, scores[1] / c, scores[2] / c]

#pointer_gen_model.


