import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import scalable.persistence as tree_io
from scalable.algorithm.nodeharvest import NodeHarvest
from scalable.modeldump import dumpBoostingTrees
from scalable.utils import get_trained_model
from sklearn.metrics import accuracy_score
from scalable.model_utils import ModelUtil

import numpy as np

config = json.loads(open("config.json", "r").read())
fname = "result/nodeharvest.txt"
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
            if k in existing_result or config["multiclass"][di] == 1:
            #if config["multiclass"][di] == 1:
                print('passed')
                continue
            max_nodecount = n * 1.5
            model = get_trained_model(data_name, model_name)
            original_accuracy, prec, f1 = model.get_performance()
            original_accuracy = round(original_accuracy, 4)
            x_train = model.X_train
            y_train = (model.clf.predict(x_train) - 0.5) * model.parameters['n_estimators'] * 2

            x_test = model.X_test
            y_pred = model.clf.predict(x_test)

            if model_name != 'lightgbm':
                nh = NodeHarvest(solver='cvx_robust', verbose=True, max_nodecount=max_nodecount)
                nh.fit(model.clf, x_train, y_train)
                n_nodes = nh.coverage_matrix_.shape[1]
                weight = nh.get_weights()
                n_selected = np.sum(weight > 0)
                y_est = nh.predict(x_test)
                y_est = np.where(y_est > 0, 1, 0)
            else:
                dumpBoostingTrees(model, '/tmp/tmp.tree')
                clf = tree_io.regressor_from_file('/tmp/tmp.tree', x_train, y_train, pruning=True, num_trees=model.parameters['n_estimators'])
                nh = NodeHarvest(solver='cvx_robust', verbose=True, max_nodecount=max_nodecount)
                nh.fit(clf, x_train, y_train)
                n_nodes = nh.coverage_matrix_.shape[1]
                weight = nh.get_weights()
                n_selected = np.sum(weight > 0)
                y_est = nh.predict(x_test)
                y_est = np.where(y_est > 0, 1, 0)
                

            model.generate_path()
            selected = np.flatnonzero(weight)
            paths = [model.paths[i] for i in selected]
            utils = ModelUtil(data_name, model_name)
            utils.init_anomaly_detection()
            score = utils.anomaly_score(paths)
            score = round(np.mean(score), 4)
            accuracy = accuracy_score(y_est, model.y_test)
            fidelity_test = accuracy_score(y_est, y_pred)
            if accuracy < 0.5:
                accuracy = 1 - accuracy
                fidelity_test = 1 - fidelity_test

            accuracy = round(accuracy, 4)
            fidelity_test = round(fidelity_test, 4)
            n_selected = int(n_selected)
            ret = {
                'dataset': data_name,
                'model': model_name,
                'original_accuracy': original_accuracy,
                'surrogate_accuracy': accuracy,
                'fidelity': fidelity_test,
                'anomaly_score': score,
                'actual_rules': n_selected,
                'expected_rules': n,
            }

            print(f'fidelity: {round(fidelity_test, 4)}')
            f = open(fname, 'a')
            f.write(json.dumps(ret) + '\n')
            f.close()
