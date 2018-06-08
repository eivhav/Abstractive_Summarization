
import tensorflow as tf

event_path = "/home/havikbot/MasterThesis/runs/DataSampling/May19_21-34-49_x99DM_CNN_50k_Coverage_DecoderAttn_MLE_tf=85_lr=1e3_max75_NoveltySample95d005"

event_file = "events.out.tfevents.1526758489.x99"

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


keys = ['4-Beam/Rouge_l_perl', '4-Beam/Rouge_1_perl', '4-Beam/Rouge_2_perl', '4-Beam/Rouge_3_perl', '4-Beam/Tri_novelty',
        '4-Beam/p_gens']

#print("\t".join([str(best_model[0])]+[str(round(log[best_model[0]][k], 3)) for k in Keys]))

graph_values = dict()
for k in keys: graph_values[k] = [-1]
for i in log.keys():
    for k in keys:
        if log[i][k] > graph_values[k][-1]: graph_values[k][-1] = log[i][k]
        if i % 5000 == 0: graph_values[k].append(-1)


def produce_cordinates(path, file, interval=5000, metric='4-Beam/Rouge_l_perl'):
    log = dict()

    for e in tf.train.summary_iterator(path + "/" + file):
        if e.step % 1000 == 0 and e.step != 0:
            if e.step not in log: log[e.step] = dict()
            for v in e.summary.value:
                log[e.step][v.tag] = v.simple_value

    graph_values = dict()
    for k in keys: graph_values[k] = [[interval, -1]]
    for i in log.keys():
        for k in keys:
            if log[i][k] > graph_values[k][-1][1]: graph_values[k][-1][1] = log[i][k]
            if i % interval == 0: graph_values[k].append([i+interval, -1])


    print("".join(["("+str(v[0]/1000)+","+str(round(v[1], 2))+")" for v in graph_values[metric]]))

