
import tensorflow as tf

event_path = "/home/havikbot/MasterThesis/runs/Baseline/NoDecoderAttn/May22_21-18-55_samuel03DM_CNN_50k_realBilTemp_noDecoderAttn_tf=85_lr=1e3_max75"

event_file = "events.out.tfevents.1527016735.samuel03"

log = dict()

for e in tf.train.summary_iterator(event_path + "/"+event_file):
    if e.step % 1000 == 0 and e.step != 0:
        if e.step not in log: log[e.step] = dict()
        for v in e.summary.value:
            log[e.step][v.tag] = v.simple_value

best_model = [0, -1]
for step in log.keys():
    if '4-Beam/Rouge_l_perl' not in log[step]: continue
    rouge_l = log[step]['4-Beam/Rouge_l_perl']
    if rouge_l > best_model[1]:
        best_model[1] = rouge_l
        best_model[0] = step


Keys = ['4-Beam/Rouge_l_perl', '4-Beam/Rouge_1_perl', '4-Beam/Rouge_2_perl', '4-Beam/Rouge_3_perl', '4-Beam/Tri_novelty',
        '4-Beam/p_gens']

print("\t".join([str(best_model[0])]+[str(round(log[best_model[0]][k], 3)) for k in Keys]))