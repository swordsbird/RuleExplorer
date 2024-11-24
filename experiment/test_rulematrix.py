import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
from scalable.utils import get_trained_model
from scalable.model_utils import ModelUtil
from rulematrix.surrogate import rule_surrogate
import numpy as np

random_state = 10

def train_surrogate(model, sampling_rate=2.0, n_rules=20, **kwargs):
    is_continuous = np.ones(X_train.shape[1]) > 0
    is_categorical = np.zeros(X_train.shape[1]) > 0
    for i in range(X_train.shape[1]):
        if len(np.unique(X_train[:, i])) <= 2:
            is_categorical[i] = True
            is_continuous[i] = False
    surrogate = rule_surrogate(model.predict,
                               X_train,
                               sampling_rate=sampling_rate,
                               is_continuous=is_continuous,
                               is_categorical=is_categorical,
                               is_integer=None,
                               number_of_rules=n_rules,
                               **kwargs)

    test_fidelity = surrogate.score(X_test)
    test_pred = surrogate.student.predict(X_test)
    test_accuracy = np.sum(test_pred == y_test) / len(y_test)
    return surrogate, test_accuracy, test_fidelity

config = json.loads(open("config.json", "r").read())
fname = "result/rulematrix.txt"
max_trail = 5
existing_result = {}
if os.path.exists(fname):
    f = open(fname, "r").read().split('\n')
    for line in f:
        if len(line) == 0:
            continue
        x = json.loads(line)
        k = f"{x['dataset']}-{x['model']}-{x['expected_rules']}"
        existing_result[k] = existing_result.get(k, 0) + 1
        if existing_result[k] > max_trail:
            max_trail = existing_result[k]

def interpret_rule(r):
    r = r[8:]
    items, pred = r.split('THEN')
    items = [t.strip()[1:-1] for t in items.split('AND')]
    conds = {}
    for t in items:
        if ' in ' in t:
            f, r = t.split(' in ')
            r0, r1 = r[1: -1].split(',')
            r0 = r0.strip()
            r1 = r1.strip()
            if r0 == '-inf':
                r0 = -1e14
            if r1 == 'inf':
                r1 = 1e14
            conds[int(f[1:])] = [float(r0), float(r1)]
        else:
            f, r = t.split(' = ')
            r = float(r)
            conds[int(f[1:])] = [r - 1e-4, r +  1e-4]

    pred = pred[8:-1].split(',')
    pred = [float(t) for t in pred]
    pred = np.argmax(pred)
    ret = {
        'range': conds,
        'output': pred,
    }
    return ret

def interpret_rulelist(surrogate):
 return [interpret_rule(l) for l in surrogate.student.__str__().split('\n') if 'IF' in l]

for n in config["number_of_rules"]:
    for data_name in config["dataset"]:
        for model_name in config["model"]:
            k = f'{data_name}-{model_name}-{n}'
            print(k)
            if existing_result.get(k, 0) == max_trail:
                print('passed')
                continue
            else:
                for it in range(max_trail - existing_result.get(k, 0)):
                    model = get_trained_model(data_name, model_name)
                    original_accuracy, prec, f1 = model.get_performance()
                    original_accuracy = round(original_accuracy, 4)
                    X_train = np.nan_to_num(model.X_train)
                    y_train = model.y_train
                    X_test = np.nan_to_num(model.X_test)
                    y_test = model.y_test
                    clf = model.clf

                    sampling_rate = 4
                    if len(X_train) > 5000 or data_name == 'abalone':
                        sampling_rate = -1

                    surrogate, accuracy, fidelity_test = train_surrogate(clf, sampling_rate, n, seed=random_state)
                    accuracy = round(accuracy, 4)
                    fidelity_test = round(fidelity_test, 4)
                    paths = interpret_rulelist(surrogate)
                    utils = ModelUtil(data_name, model_name)
                    utils.init_anomaly_detection()
                    score = utils.anomaly_score(paths)
                    all_score = [round(s, 4) for s in score]
                    #for i, p in enumerate(paths):
                    #    print(utils.interpret_path(p, to_text=True), all_score[i])
                    score = round(np.mean(score), 4)

                    ret = {
                        'dataset': data_name,
                        'model': model_name,
                        'original_accuracy': original_accuracy,
                        'surrogate_accuracy': accuracy,
                        'fidelity': fidelity_test,
                        'anomaly_score': score,
                        #'all_score': all_score,
                        'actual_rules': len(surrogate.student.rule_list),
                        'expected_rules': n,
                    }

                    print(f'fidelity: {round(fidelity_test, 4)}')
                    f = open(fname, 'a')
                    f.write(json.dumps(ret) + '\n')
                    f.close()
