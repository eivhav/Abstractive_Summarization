
import tensorflow as tf
import json

event_path = "/home/havikbot/MasterThesis/runs/DataSampling/May25_12-50-57_psvyti6phDM_CNN_50k_Coverage_Decodertf=85_lr=1e3_max75_sampling090_001_02"

event_file = "events.out.tfevents.1527267057.psvyti6ph"

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

print("\t".join([str(best_model[0])]+[str(round(log[best_model[0]][k], 3)) for k in keys]))

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


def produce_rouge_dist(models, baseline):

    results = {}
    baseline_values = {}
    for r_type in baseline['scores']['rouge_dist'].keys():
        results[r_type] = []
        baseline_values[r_type] = [v[0] for v in baseline['scores']['rouge_dist'][r_type]]
        for i in range(21): results[r_type].append([])

    for k in models.keys():
        m = models[k]
        print(k)
        for r_type in m['scores']['rouge_dist'].keys():
            for i in range(3, len(results[r_type]), 2):
                v1 = m['scores']['rouge_dist'][r_type][i][0] - baseline['scores']['rouge_dist'][r_type][i][0]
                v2 = m['scores']['rouge_dist'][r_type][i+1][0] - baseline['scores']['rouge_dist'][r_type][i+1][0]
                results[r_type][i].append((v1+v2) / 2)

    for r in results:
        print(r)
        for i in range(3, 21, 2):
            print(str(i/20)+", " + ", ".join([str(round(100*v, 2)) for v in results[r][i]]))


import json
path = '/home/havikbot/MasterThesis/best_models/RL/Final/results/'
file = path + "resultsRL.json"
baseline_file = path + 'resultsBase.json'
models_data = json.loads(open(file).read())


keys = [
'checkpoint_DM_CNN_50k_Coverage_tf=85_lr=adam5e5b25_SC0995_1xTrigram1xPerl-L_neg_ep@45000_loss@0.pickle',
'checkpoint_DM_CNN_50k_Coverage_tf=85_lr=adam5e5b25_SC0995_2_5xTrigram1xPerl-L_neg_ep@45500_loss@0.pickle',
'checkpoint_DM_CNN_50k_Coverage_tf=85_lr=adam5e5b25_SC0995_7_5xTrigram1xPerl-L_neg_ep@27000_loss@0.pickle',
]
models_data = {k: models_data[k] for k in keys}

baseline_data = json.loads(open(baseline_file).read())
baseline = baseline_data[list(baseline_data.keys())[0]]
produce_rouge_dist(models_data, baseline)


def produce_graph_table_from_tensorboard(runs, intervall=3000, _range=120000, tag='4-Beam/Rouge_l_perl'):

    logs = []
    for run in runs:
        log = dict()
        maxium = 0
        for e in tf.train.summary_iterator(run):
            if e.step % 1000 == 0:
                if e.step not in log: log[e.step] = dict()
                if e.step > maxium: maxium = e.step
                for v in e.summary.value:
                    log[e.step][v.tag] = v.simple_value

        for i in range(maxium, _range, 1000):
            log[i] = {t: -1 for t in log[maxium]}

        logs.append(log)

    print("iter, "+", ".join(r.split("max75_")[1].split("/events")[0] for r in runs))
    for i in range(0, _range-2000, intervall):
        print(str(i/1000) + ", " + ", ".join([str(round(max(log[i][tag], log[i+1000][tag], log[i+1000][tag]) , 3)) for log in logs]))


path = '/home/havikbot/MasterThesis/runs/Novelty_loss/best/'
runs = {
    path + 'Jun06_23-35-20_x99DM_CNN_50k_Coverage_tf=85_lr=adagrad5e4_max75_pgen_loss_002/events.out.tfevents.1528320920.x99',
    path + 'Jun06_23-28-04_x99DM_CNN_50k_Coverage_tf=85_lr=adagrad5e4_max75_pgen_loss_005/events.out.tfevents.1528320484.x99'
}
produce_graph_table_from_tensorboard(runs)





