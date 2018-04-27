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

model_id = 'DM_CNN_50k_Coverage_DecoderAttn_novelty=02_start_test_RL'

config = {'model_type': 'Combo',
          'embedding_size': 128, 'hidden_size': 256,
          'temporal_att': False, 'bilinear_attn': False,
          'decoder_att': True, 'input_in_pgen': True,
          'input_length': 400, 'target_length': 50,
          'model_path': current+ model_path, 'model_id': model_id }


pointer_gen_model = PGModel(config=config, vocab=data_loader.vocab, use_cuda=use_cuda)

'''
pointer_gen_model.load_model(file_path='/home/havikbot/MasterThesis/Models/',
                              file_name='checkpoint_DM_CNN_50k_Coverage_DecoderAttn_ep@45000_loss@0.pickle')

'''

#reward_module = RougePerlVersion(rouge_variants=["ROUGE-L-F"])
reward_module = RougeSumeval(rouge_variants=["ROUGE-2-F", "ROUGE-L-F"], remove_stopwords=False, stem=False)
trainer = SelfCriticalTrainer(model=pointer_gen_model, tag=model_id, rl_lambda=0.98,
                              reward_module=reward_module, reward_min=-1)

#trainer = MLE_Novelty_Trainer(pointer_gen_model, model_id. novelty_loss)



trainer.train(data_loader=data_loader, nb_epochs=25, batch_size=16,
                        optimizer=torch.optim.Adam, lr=0.0001,
                        tf_ratio=1.0, stop_criterion=None,
                        use_cuda=True, print_evry=100, start_iter=1
                        )



