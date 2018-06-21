

from __future__ import unicode_literals, print_function, division
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from Models.model import *
from Training.MLE_trainers import *
from Data.data_loader import *
from Evaluation.scorer import *
from Training.RL_rewards import *
use_cuda = torch.cuda.is_available()


samuel = '/srv/'
x99 = '/home/'
paperspace = 'home/paperspace/'
current = x99
sys.path.append(current + 'havikbot/MasterThesis/Code/Abstractive_Summarization/')

path = '/home/havikbot/MasterThesis/best_models/RL/Final/'
file = 'checkpoint_DM_CNN_50k_Coverage_tf=85_lr=adam5e5b25_SC0995_20xBigram1xPerl-L_neg_ep@19500_loss@0.pickle'
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

scorer.rewards_to_score = {
        "bi_gram_reward": TrigramNovelty(remove_stopwords=False, stem=False, method='bi_gram'),
        "tri_gram_reward": TrigramNovelty(remove_stopwords=False, stem=False, method='tri_gram')
    }

test_batches = data_loader.load_data('test', batch_size=25)
scores = scorer.score_model(test_batches[4:5], use_cuda, beam=5, verbose=True, rouge_dist=False)

keys = ['Rouge_l_perl', 'Rouge_1_perl', 'Rouge_2_perl', 'Rouge_3_perl', 'Tri_novelty', 'p_gens']
print("\t".join([str(round(scores[k], 3)) for k in keys]))




trainer = MLE_Novelty_Trainer(pointer_gen_model, "", novelty_lambda=-1)
trainer.model.encoder_optimizer = torch.optim.Adagrad(pointer_gen_model.encoder.parameters(), lr=1)
trainer.model.decoder_optimizer = torch.optim.Adagrad(pointer_gen_model.decoder.parameters(), lr=1)
trainer.model.criterion = nn.NLLLoss()
loss_values = dict()
for batch in data_loader.load_data('val', batch_size=10)[:50]:
    _time, losses = trainer.train_batch(batch, use_cuda, backprop=False)
    if len(loss_values) == 0:
        for k in losses: loss_values[k] = 0
    for k in losses: loss_values[k] += losses[k]



