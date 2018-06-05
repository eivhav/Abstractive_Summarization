from __future__ import unicode_literals, print_function, division

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

samuel = '/srv/'
x99 = '/home/'
paperspace = 'home/paperspace/'
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
best_model_path = 'havikbot/MasterThesis/best_models/Coverage_base/'
model_path = 'havikbot/MasterThesis/Models/'


data_loader = DataLoader(data_path, 45, None, s_damp=0.2)

# 'TemporalAttn' or CoverageAttn

#model_id = 'DM_CNN_50k_Coverage_tf=85_lr=adagrad5e4_max75_att_loss_005'
model_id = 'DM_CNN_50k_Coverage_tf=85_lr=grad1e4_SC1_alt(bleu)_max45'

config = {'model_type': 'Combo',
          'embedding_size': 128, 'hidden_size': 256,
          'temporal_att': False, 'bilinear_attn': False,
          'decoder_att': True, 'input_in_pgen': True,
          'input_length': 400, 'target_length': 76,
          'model_path': current+ model_path, 'model_id': model_id }


pointer_gen_model = PGModel(config=config, vocab=data_loader.vocab, use_cuda=use_cuda,
                            model_file=current+ best_model_path+"checkpoint_DM_CNN_50k_TempBil_MLE_tf=85_lr=e3_max75_ep@79000_loss@0.pickle" )

'''
pointer_gen_model.load_model(file_path='/home/havikbot/MasterThesis/Models/',
            file_name='checkpoint_DM_CNN_50k_cont_Coverage_DecoderAttn_SC=1_PerlL2_pos_rws_ep@17400_loss@0.pickle')

'''
#reward_module = RougePerlVersion(rouge_variants=["ROUGE-L-F", "ROUGE-2-F"], nb_workers=8)
#reward_module = RougeSumeval(rouge_variants=["ROUGE-2-F", "ROUGE-L-F"], remove_stopwords=True, stem=False)
#reward_module = TrigramNovelty(remove_stopwords=False, stem=False)
#reward_module = SacreBleu()
reward_module = AlternateRewards([SacreBleu()])
                                 #RougeSumeval(rouge_variants=["ROUGE-2-F", "ROUGE-L-F"], remove_stopwords=False, stem=True)]


trainer = SelfCriticalTrainer(model=pointer_gen_model, tag=model_id, rl_lambda=0, reward_module=reward_module, reward_min=0)


#trainer = MLE_Novelty_Trainer(pointer_gen_model, model_id, novelty_lambda=0.05)
#trainer.novelty_loss_type = 'att'
#trainer.beam_batch_size = 15


trainer.train(data_loader=data_loader, nb_epochs=25, batch_size=10,
                        optimizer=torch.optim.Adagrad, lr=0.0000001,
                        tf_ratio=0.85, stop_criterion=None,
                        use_cuda=True, print_evry=500, start_iter=0, n_sampling_decay=None, new_optm=True
                        )



