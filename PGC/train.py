from __future__ import unicode_literals, print_function, division


import torch
import pickle
from model import PGCModel
from data_loader import *

use_cuda = torch.cuda.is_available()

data_path = '/home/havikbot/MasterThesis/Data/CNN_dailyMail/DailyMail/model_datasets/'
path_2 = '/home/shomea/h/havikbot/MasterThesis/'

model_path = '/home/shomea/h/havikbot/MasterThesis/Models/PGC_v2/'


with open(path_2 +'DM_25k_summary_v2.pickle', 'rb') as f: dataset = pickle.load(f)
training_pairs = dataset.summary_pairs[0:int(len(dataset.summary_pairs)*0.8)]
test_pairs = dataset.summary_pairs[int(len(dataset.summary_pairs)*0.8):]

config = {'embedding_size': 128, 'hidden_size': 256, 'input_length': 300, 'target_length': 40}


pointer_gen_model = PGCModel(config=config, vocab=dataset.vocab, use_cuda=use_cuda, model_path=model_path, model_id='test')


'''
pointer_gen_model.train(data=training_pairs, val_data=test_pairs,
                        nb_epochs=20, batch_size=32,
                        optimizer=torch.optim.Adagrad, lr=0.015,
                        tf_ratio=0.5, stop_criterion=None,
                        use_cuda=True, _print=True
                        )

'''
pointer_gen_model.load_model(model_path, 'checkpoint_test_ep@.pickle')

for sample in test_pairs[20:25]:
    print(sample.source_text)
    print()
    print(sample.target_text)
    print()
    result = pointer_gen_model.predict([sample], target_length=50, beam_size=5, use_cuda=use_cuda)
    for r in result: print(r)
    print('\n\n')
