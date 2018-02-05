from __future__ import unicode_literals, print_function, division


import torch
import pickle
from deepRL_model import PGCModel
#from PGC.model import PGCModel
from data_loader import *

use_cuda = torch.cuda.is_available()

data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/Data/'
path_3 = '/home/havikbot/MasterThesis/Data/'
path4 = '/home/havikbot/MasterThesis/Data/NYTcorpus/with_abstract/model_data/'
samuel_path = '/srv/havikbot/MasterThesis/Data/'

model_path = '/home/shomea/h/havikbot/MasterThesis/Models/DeepRL/'
data_set_name_dm = 'DM_25k_summary_v2.pickle'
data_set_name_nyt = 'NYT_40k_summary_v3.pickle'


with open(samuel_path +data_set_name_nyt, 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.8)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

config = {'embedding_size': 200, 'hidden_size': 400, 'input_length': 300, 'target_length': 50}


pointer_gen_model = PGCModel(config=config, vocab=dataset.vocab, use_cuda=use_cuda,
                             model_path=model_path, model_id='DeepRL')
'''
pointer_gen_model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=25, batch_size=50,
                        optimizer=torch.optim.Adam, lr=0.001,
                        tf_ratio=0.5, stop_criterion=None,
                        use_cuda=True, _print=True
                        )

'''


pointer_gen_model.load_model(file_path='/srv/havikbot/MasterThesis/Models/',
                              file_name='checkpoint_DeepRL_nyt_ep@.pickle')


for i in range(140, 150):
    pair = training_pairs[i]
    pred = pointer_gen_model.predict([pair], 75, False, use_cuda)
    pred_beam = pointer_gen_model.predict([pair], 75, 5, use_cuda)
    print("True: ", pair.get_text(pair.full_target_tokens, pointer_gen_model.vocab))
    print("Pred:", " ".join([t[0]['word'] for t in pred]))
    print("Pred:", " ".join([t[0]['word']+"{"+str(round(t[0]['p_gen'], 1)) +"}"  for t in pred]))
    for i in range(len(pred_beam)): print("Beam", i, pred_beam[i])
    print()

pointer_gen_model.save_model(path='/srv/havikbot/MasterThesis/Models/', id ='DeepRL_valid', loss=-1, epoch=-1)