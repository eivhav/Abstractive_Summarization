from __future__ import unicode_literals, print_function, division

import sys
import os
from Models.model import *
from Training.MLE_trainers import *
from Data.data_loader import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"

samuel = '/srv/'
x99 = '/home/'
current = x99

sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/')
use_cuda = torch.cuda.is_available()

data_path = '/home/havikbot/MasterThesis/Data/Model_data/CNN_DM/'
data_loader = DataLoader(data_path)

file_path='/home/havikbot/MasterThesis/Model compare/11_may_tempBIl/'
#file_name='checkpoint_DM_CNN_50k_Coverage_MLE_tf=85_lr=e3_max75_ep@84000_loss@0.pickle'
model_path = 'havikbot/MasterThesis/Models/'

config = {'model_type': 'Combo',
          'embedding_size': 128, 'hidden_size': 256,
          'temporal_att': False, 'bilinear_attn': False,
          'decoder_att': True, 'input_in_pgen': True,
          'input_length': 400, 'target_length': 76,
          'model_path': current+ model_path, 'model_id': ""}

import glob
scores = []
for file_name in sorted(list(glob.iglob(file_path+"*.pickle"))):

    pointer_gen_model = PGModel(config=config, vocab=data_loader.vocab, use_cuda=use_cuda,
                                model_file=file_name)
    trainer = Trainer(model=pointer_gen_model, tag="")

    test_batches = data_loader.load_data('test', batch_size=10)
    scores.append(trainer.score_model(test_batches[0:100], use_cuda, beam=5, verbose=True))










#scores = trainer.score_model(test_batches, use_cuda, beam=5, verbose=True)
print(scores)


beams = pointer_gen_model.predict_v2(test_batches[0], 60, 5, True)
pred_seq = beams[1][0][0].split(" ")[10:25]
p_gens = beams[1][0][-2]
source = " ".join(test_batches[0][1].source[0:15])
att = beams[1][0][-1].data[0].cpu().narrow(0, 10, 15).narrow(1,0,15)

import Testing.attention_viz as viz
viz.showAttention(source, pred_seq, att)

import numpy as np
scaled_att = np.array(p_gens[10:25])



scores = trainer.score_model(test_batches, use_cuda, beam=5, verbose=True)
print(scores)


'''
{'Rouge_1': 36.874586695611306, 'Rouge_2': 16.65283348357296, 'Rouge_L': 23.368729027497835, 
'Tri_novelty': 0.17834037835730515, 
'Rouge_1_perl': 38.339, 'Rouge_2_perl': 17.138, 'Rouge_3_perl': 9.577, 'Rouge_l_perl': 35.894}


{'Rouge_1': 38.49024989239435,
 'Rouge_1_perl': 40.172000000000004,
 'Rouge_2': 17.53878776286983,
 'Rouge_2_perl': 18.209,
 'Rouge_3_perl': 10.517999999999999,
 'Rouge_L': 24.998178427328885,
 'Rouge_l_perl': 37.769999999999996,
 'Tri_novelty': 0.1609169732818691}


'''