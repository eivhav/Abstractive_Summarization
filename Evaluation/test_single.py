

from __future__ import unicode_literals, print_function, division
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from Models.model import *
from Training.MLE_trainers import *
from Data.data_loader import *
from Evaluation.scorer import *
use_cuda = torch.cuda.is_available()


samuel = '/srv/'
x99 = '/home/'
paperspace = 'home/paperspace/'
current = x99
sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/')

path = '/home/havikbot/MasterThesis/Models/'
file = 'checkpoint_DM_CNN_50k_Coverage_tf=85_lr=adam1e4_SC1_Bigram_novelty_neg_ep@23000_loss@0.pickle'
data_path = '/home/havikbot/MasterThesis/Data/Model_data/CNN_DM/'

config =         {'model_type': 'Combo',
            'embedding_size': 128, 'hidden_size': 256,
            'temporal_att': False, 'bilinear_attn': False,
            'decoder_att': True, 'input_in_pgen': True,
            'input_length': 400, 'target_length': 76,
            'model_path': "", 'model_id': "" }

data_loader = DataLoader(data_path, 75, None, s_damp=0.2)
pointer_gen_model = PGModel(config=config, vocab=data_loader.vocab, use_cuda=use_cuda,
                            model_file=path+file)

scorer = Scorer(model=pointer_gen_model)
test_batches = data_loader.load_data('test', batch_size=15)
scores = scorer.score_model(test_batches[0:100], use_cuda, beam=5, verbose=True)

keys = ['Rouge_l_perl', 'Rouge_1_perl', 'Rouge_2_perl', 'Rouge_3_perl', 'Tri_novelty', 'p_gens']
print("\t".join([str(round(scores[k], 3)) for k in keys]))

