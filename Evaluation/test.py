from __future__ import unicode_literals, print_function, division

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

samuel = '/srv/'
x99 = '/home/'
current = x99

sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/')

from Models.model import *
#from Training.MLE_training import *
from Training.RL_trainers import *
from Training.RL_rewards import *
from Training.MLE_trainers import *

from Data.data_loader import *

use_cuda = torch.cuda.is_available()

data_path = '/home/havikbot/MasterThesis/Data/Model_data/CNN_DM/'
model_path = 'havikbot/MasterThesis/Models/'


data_loader = DataLoader(data_path)

# 'TemporalAttn' or CoverageAttn

model_id = 'DM_CNN_50k_cont_Coverage_DecoderAttn_SC=1_PerlL2_pos_rws'

config = {'model_type': 'Combo',
          'embedding_size': 128, 'hidden_size': 256,
          'temporal_att': False, 'bilinear_attn': False,
          'decoder_att': True, 'input_in_pgen': True,
          'input_length': 400, 'target_length': 50,
          'model_path': current+ model_path, 'model_id': model_id }


pointer_gen_model = PGModel(config=config, vocab=data_loader.vocab, use_cuda=use_cuda)


pointer_gen_model.load_model(file_path='/home/havikbot/MasterThesis/Models/',
            file_name='checkpoint_DM_CNN_50k_cont_Coverage_DecoderAttn_SC=1_PerlL2_pos_rws_ep@16800_loss@0.pickle')

trainer = Trainer(model=pointer_gen_model, tag=model_id)

test_batches = data_loader.load_data('test', batch_size=1)

scores = trainer.score_model(test_batches, use_cuda, beam=5, verbose=True)
print(scores)


'''
{'Rouge_1': 36.874586695611306, 'Rouge_2': 16.65283348357296, 'Rouge_L': 23.368729027497835, 
'Tri_novelty': 0.17834037835730515, 
'Rouge_1_perl': 38.339, 'Rouge_2_perl': 17.138, 'Rouge_3_perl': 9.577, 'Rouge_l_perl': 35.894}


'''