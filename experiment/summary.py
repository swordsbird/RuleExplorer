import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import numpy as np

methods = [
    { 'name': 'Our', 'path': 'result/our.txt'},
    { 'name': 'Rule Matrix', 'path': 'result/rulematrix.txt'},
    { 'name': 'HSR', 'path': 'result/hsr.txt'},
    { 'name': 'Node Harvest', 'path': 'result/nodeharvest.txt'}
]
metrics = ['fidelity', 'anomaly_score']
expected_rules = 80

config = json.loads(open("config.json", "r").read())
fname = "result/rulematrix.txt"

result = {}
tex_prefix = {
    'credit-RF': '\multirow{2}*{Credit Card~\cite{australian_credit_approval}} & \multirow{2}*{2} & \multirow{2}*{690} & ',
    'wine-RF': '\multirow{2}*{Wine Quality~\cite{wine_quality}} & \multirow{2}*{2} & \multirow{2}*{1,599} & ',
    'crime-RF': '\multirow{2}*{Crime~\cite{crime}} & \multirow{2}*{2} & \multirow{2}*{1,994} & ',
    'abalone-RF': '\multirow{2}*{Abalone~\cite{abalone}} & \multirow{2}*{4} & \multirow{2}*{4,177} & ',
    'obesity-RF': '\multirow{2}*{Obesity~\cite{obesity_levels}} & \multirow{2}*{7} & \multirow{2}*{2,111} & ',
    'drybean-RF': '\multirow{2}*{Dry Bean~\cite{dry_bean}} & \multirow{2}*{7} & \multirow{2}*{13,611} & '
}

for method in methods:
    name = method['name']
    path = method['path']
    if os.path.exists(method['path']):
        f = open(method['path'], "r").read().split('\n')
        for line in f:
            if len(line) == 0:
                continue
            x = json.loads(line)
            if x['expected_rules'] != expected_rules:
                continue
            k = f"{x['dataset']}-{x['model']}"
            if k not in result:
                result[k] = {}
            if name not in result[k]:
                result[k][name] = {}
                for j in metrics:
                    result[k][name][j] = []
            for j in metrics:
                result[k][name][j].append(x.get(j, -1))

fname = 'result/summary.txt'
f = open(fname, 'w')
for di, data_name in enumerate(config["dataset"]):
    # f.write(data_name + '\n')
    for model_name in config["model"]:
        k = f'{data_name}-{model_name}'
        if model_name == 'random forest':
            model = 'RF'
        elif model_name == 'lightgbm':
            model = 'GBT'
        else:
            model = 'N/A'
        all_values = [[] for _ in methods]
        for j in metrics:
            values = []
            for method in methods:
                name = method['name']
                if name not in result[k]:
                    values.append(-1)
                else:
                    values.append(np.mean(result[k][name][j]))
            best = np.argmax(values)
            values = [format(v, '.4f') if v != -1 else 'N/A' for v in values]
            values[best] = '\\textbf{' + values[best] + '}'
            for i in range(len(methods)):
                all_values[i].append(values[i])
        values = []
        for v in all_values:
            values += v
        values = [model] + values
        prefix = tex_prefix.get(f'{data_name}-{model}', '~ & ~ & ~ & ')
        table_body = ' & '.join(values)
        line = prefix + table_body + '\\\\\n'
        f.write(line)
    if di + 1 < len(config['dataset']):
        f.write('\\hline\n')
f.close()
