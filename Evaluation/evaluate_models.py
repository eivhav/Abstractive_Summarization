
from __future__ import unicode_literals, print_function, division

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

samuel = '/srv/'
x99 = '/home/'
paperspace = 'home/paperspace/'
current = x99

sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/')

from Models.model import *
from Training.MLE_trainers import *

from Data.data_loader import *
from Evaluation.scorer import *

use_cuda = torch.cuda.is_available()

data_path = 'havikbot/MasterThesis/Data/Model_data/CNN_DM/'
model_path = 'havikbot/MasterThesis/best_models/RL/Final/'


models = {

    'checkpoint_DM_CNN_50k_Coverage_tf=85_lr=adam5e5b25_SC0995_Perl-L_negv2_ep@29500_loss@0.pickle':
        {'model_type': 'Combo',
            'embedding_size': 128, 'hidden_size': 256,
            'temporal_att': False, 'bilinear_attn': False,
            'decoder_att': True, 'input_in_pgen': True,
            'input_length': 400, 'target_length': 76,
            'model_path': current+ model_path, 'model_id': "" }



}


data_loader = DataLoader(current + data_path, 75, None, s_damp=0.2)
b_size = 10
n_batches = 1500

# 'TemporalAttn' or CoverageAttn

from Training.RL_rewards import TrigramNovelty
novelty_reward = TrigramNovelty()

results = dict()
for file_name in models.keys():
    results[file_name] = dict()
    config = models[file_name]
    pointer_gen_model = PGModel(config=config, vocab=data_loader.vocab, use_cuda=use_cuda,
                                model_file=current +model_path + file_name)
    scorer = Scorer(model=pointer_gen_model)
    scorer.beam_batch_size = b_size
    scorer.rewards_to_score = {
        "bi_gram_reward": TrigramNovelty(remove_stopwords=False, stem=False, method='bi_gram'),
        "tri_gram_reward": TrigramNovelty(remove_stopwords=False, stem=False, method='tri_gram')
    }
    test_batches = data_loader.load_data('test', batch_size=b_size)

    scores = scorer.score_model(test_batches[0:n_batches], use_cuda, beam=5, verbose=True, rouge_dist=True)
    results[file_name]['scores'] = scores
    keys = ['Rouge_l_perl', 'Rouge_1_perl', 'Rouge_2_perl', 'Rouge_3_perl', 'Tri_novelty', 'p_gens']
    print(file_name)
    print("\t".join([str(round(scores[k], 3)) for k in keys]))
    results[file_name]['print_scores'] = "\t".join([str(round(scores[k], 3)) for k in keys])

    preds_beam = pointer_gen_model.predict_v2(test_batches[0], 100, 5, use_cuda)
    refs = [pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab).split(" EOS")[0] for pair in test_batches[0]]
    for i in range(b_size):
        results[file_name][i] = preds_beam[i][0][0]

import json
sample_preds = []
for i in range(b_size):
    sample_preds.append({m: results[m][i] for m in results.keys()})


with open(current +model_path + "resultsBase.json", "w") as handle: handle.write(json.dumps(results))
with open(current +model_path + "sample_predsBase.json", "w") as handle: handle.write(json.dumps(sample_preds))











