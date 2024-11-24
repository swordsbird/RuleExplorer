import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import pandas as pd
from scalable.utils import get_trained_model
from surrogate_rule import forest_info
from surrogate_rule import tree_node_info
from scalable.model_utils import ModelUtil

import numpy as np

filter_threshold = {
    "support": 5,
    "fidelity": .85,
    "num_feat": 4,
    "num_bin": 3,
}
num_bin = filter_threshold['num_bin']
random_state = 10

def extract_rules_from_RF(model):
    X = model.X_train
    y_gt = model.y_train.tolist()
    y_pred = model.clf.predict(model.X_train).tolist()
    y_gt = np.array(y_gt)
    y_pred = np.array(y_pred)
    columns = model.features
    if len(columns) < X.shape[1]:
        for i in range(X.shape[1] - len(columns)):
            columns.append(f'new feature{i}')
    else:
        columns = columns[:X.shape[1]]
    df = pd.DataFrame(data=X, columns = columns)
    n_cls = len(model.output_labels)
    real_min = model.X_train.min(axis = 0)
    real_max = model.X_train.max(axis = 0)
    # train surrogate
    surrogate_obj = tree_node_info.tree_node_info()

    surrogate_obj.initialize(X=X, y=y_gt,
                             y_pred=y_pred, debug_class=-1,
                             attrs=columns, filter_threshold=filter_threshold,
                             n_cls=n_cls,
                             num_bin=num_bin, verbose=False
    ).train_surrogate_random_forest().tree_pruning()

    forest_obj = tree_node_info.forest()
    forest_obj.initialize(
        trees=surrogate_obj.tree_list, cate_X=surrogate_obj.cate_X,
        y=surrogate_obj.y, y_pred=surrogate_obj.y_pred, attrs=columns, num_bin=num_bin,
        real_percentiles=surrogate_obj.real_percentiles,
        real_min=surrogate_obj.real_min, real_max=surrogate_obj.real_max,
    ).construct_tree().extract_rules()

    forest = forest_info.Forest()

    forest.initialize(forest_obj.tree_node_dict, real_min, real_max, surrogate_obj.percentile_info,
        df, y_pred, y_gt,
        forest_obj.rule_lists,
        model.output_labels, 2)
    forest.initialize_rule_match_table()
    forest.initilized_rule_overlapping()
    try:
        res = forest.find_the_min_set()
    except:
        return False
    lattice = forest.get_lattice_structure(res['rules'])
    # print('rule', res['rules'])

    max_feat = 0
    min_feat = 111
    avg_feat = 0.0
    for rule in res['rules']:
        if (len(rule['rules']) > max_feat):
            max_feat = len(rule['rules'])
        if (len(rule['rules']) < min_feat):
            min_feat = len(rule['rules'])
        avg_feat += len(rule['rules'])

    rules = []
    for r in res['rules']:
        p = {}
        p['range'] = {}
        p['output'] = r['label']
        try:
            for cond in r['rules']:
                if cond['sign'] == '<=':
                    p['range'][cond['feature']] = [-1e14, cond['threshold']]
                elif cond['sign'] == '>':
                    p['range'][cond['feature']] = [cond['threshold'] + 1e-6, 1e14]
                else:
                    p['range'][cond['feature']] = [cond['threshold0'] + 1e-6, cond['threshold1']]
            rules.append(p)
        except:
            print(r)
    res['rules'] = rules
    return rules, res['coverage'], res['fidelity']

config = json.loads(open("config.json", "r").read())
fname = "result/hsr.txt"
existing_result = set()
if os.path.exists(fname):
    f = open(fname, "r").read().split('\n')
    for line in f:
        if len(line) == 0:
            continue
        x = json.loads(line)
        k = f"{x['dataset']}-{x['model']}-{x['expected_rules']}"
        existing_result.add(k)

min_fidelity_list = [.6, .65, .7, .75, .8, .85, .9]
num_feat_list = [3, 4, 5, 6]

for n in config["number_of_rules"]:
    for data_name in config["dataset"]:
        for model_name in config["model"]:
            k = f'{data_name}-{model_name}-{n}'
            print(k)
            if k in existing_result:
                print('passed')
                continue
            model = get_trained_model(data_name, model_name)
            original_accuracy, prec, f1 = model.get_performance()
            original_accuracy = round(original_accuracy, 4)
            X_train = model.X_train
            y_train = model.y_train
            X_test = model.X_test
            y_test = model.y_test
            clf = model.clf

            best_fidelity = 0
            best_min_fidelity = -1
            for min_fidelity in min_fidelity_list:
                filter_threshold['fidelity'] = min_fidelity
                paths, coverage, fidelity = extract_rules_from_RF(model)
                if fidelity > best_fidelity:
                    best_min_fidelity = min_fidelity
                    best_fidelity = fidelity
                    best_paths = paths
                    best_coverage = coverage
            filter_threshold['fidelity'] = best_min_fidelity

            best_num_feat = filter_threshold['num_feat']
            for num_feat in num_feat_list:
                filter_threshold['num_feat'] = num_feat
                paths, coverage, fidelity = extract_rules_from_RF(model)
                if fidelity > best_fidelity:
                    best_num_feat = num_feat
                    best_fidelity = fidelity
                    best_paths = paths
                    best_coverage = coverage
            filter_threshold['num_feat'] = best_num_feat
            
            fidelity = round(best_fidelity, 4)
            paths = best_paths
            n_selected = len(paths)
            utils = ModelUtil(data_name, model_name)
            utils.init_anomaly_detection()
            score = utils.anomaly_score(paths)
            score = round(np.mean(score), 4)

            ret = {
                'dataset': data_name,
                'model': model_name,
                'original_accuracy': original_accuracy,
                'fidelity': fidelity,
                'anomaly_score': score,
                'actual_rules': n_selected,
                'expected_rules': n,
            }

            print(f'fidelity: {round(fidelity, 4)}')
            f = open(fname, 'a')
            f.write(json.dumps(ret) + '\n')
            f.close()
