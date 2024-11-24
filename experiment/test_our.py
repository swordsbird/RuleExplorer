import sys
sys.path.append('.')
sys.path.append('..')

import os
from scalable.algorithm.tree_extractor import path_extractor
from scalable.algorithm.model_reduction import Extractor
from scalable.hierarchy import generate_hierarchy
import numpy as np
import json

config = json.loads(open("config.json", "r").read())
fname = "result/our.txt"
existing_result = set()
if os.path.exists(fname):
    f = open(fname, "r").read().split('\n')
    for line in f:
        if len(line) == 0:
            continue
        x = json.loads(line)
        k = f"{x['dataset']}-{x['model']}-{x['expected_rules']}"
        existing_result.add(k)

for n in config["number_of_rules"]:
    for di, data_name in enumerate(config["dataset"]):
        for model_name in config["model"]:
            k = f'{data_name}-{model_name}-{n}'
            print(k)
            if k in existing_result:
                print('passed')
                continue
            if config['multiclass'][di] == 1:
                class_weight = None#'balanced'
            else:
                class_weight = 'balanced'#None
            model, paths, info, _ = generate_hierarchy(data_name, model_name, n = n, class_weight=class_weight)
            original_accuracy, prec, f1 = model.get_performance()
            original_accuracy = round(original_accuracy, 4)
            print(f'Number of rules: {len(paths)}')
            alpha = model.parameters['n_estimators'] * n / len(paths)
            ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
            w = np.array([p['weight'] for p in paths])
            idx = np.flatnonzero(w)

            fidelity = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))
            fidelity = round(fidelity, 4)
            accuracy = ex.evaluate(w, model.X_test, model.y_test)
            accuracy = round(accuracy, 4)
            score = np.array([paths[i]['score'] for i in idx])
            score = round(np.mean(score), 4)

            xi = round(info['xi'], 4)
            lambda_ = round(info['lambda_'], 4)
            ret = {
                'dataset': data_name,
                'model': model_name,
                'original_accuracy': original_accuracy,
                'surrogate_accuracy': accuracy,
                'fidelity': fidelity,
                'anomaly_score': score,
                'actual_rules': len(idx),
                'expected_rules': n,
            }

            print(f'fidelity: {round(fidelity, 4)}')
            f = open(fname, 'a')
            f.write(json.dumps(ret) + '\n')
            f.close()

            ret['xi'] = xi
            ret['lambda_'] = lambda_
            f = open(fname + '2', 'a')
            f.write(json.dumps(ret) + '\n')
            f.close()


