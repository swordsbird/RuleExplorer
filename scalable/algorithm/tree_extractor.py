
from os import path
from copy import deepcopy
import numpy as np
from scalable.rule_query import Table

def visit_boosting_tree(tree, path = {}, missing = []):
    if 'decision_type' not in tree:
        return [{
            'range': path,
            'missing': missing,
            'value': tree['leaf_value'],
            'weight': 1,#tree.get('leaf_weight', 1),
        }]

    key = tree['split_feature']
    thres = tree['threshold']
    default_left = tree['default_left']
    ret = []
    leftpath = deepcopy(path)
    left_missing = deepcopy(missing)
    if key in leftpath:
        r = leftpath[key]
        leftpath[key] = [r[0], min(r[1], thres)]
    else:
        leftpath[key] = [-1e17, thres]
    if default_left:
        left_missing.append(key)
    ret += visit_boosting_tree(tree['left_child'], leftpath, left_missing)

    rightpath = deepcopy(path)
    right_missing = deepcopy(missing)
    if key in rightpath:
        r = rightpath[key]
        rightpath[key] = [max(r[0], thres), r[1]]
    else:
        rightpath[key] = [thres, 1e17]
    if not default_left:
        right_missing.append(key)
    ret += visit_boosting_tree(tree['right_child'], rightpath, right_missing)

    return ret

def visit_decision_tree(tree, index = 0, path = {}):
    if tree.children_left[index] == -1 and tree.children_right[index] == -1:
        return [{
            'range': path,
            'value': 0,
            'weight': 1,
        }]
    key = tree.feature[index]
    thres = tree.threshold[index]
    ret = []
    leftpath = deepcopy(path)
    if key in leftpath:
        r = leftpath[key]
        leftpath[key] = [r[0], min(r[1], thres)]
    else:
        leftpath[key] = [-1e17, thres]
    if tree.children_left[index] != index:
        ret += visit_decision_tree(tree, tree.children_left[index], leftpath)

    rightpath = deepcopy(path)
    if key in rightpath:
        r = rightpath[key]
        rightpath[key] = [max(r[0], thres), r[1]]
    else:
        rightpath[key] = [thres, 1e17]
    if tree.children_right[index] != index:
        ret += visit_decision_tree(tree, tree.children_right[index], rightpath)

    return ret

def assign_samples(paths, data):
    X, y = data
    for path in paths:
        ans = 2 * y - 1
        m = path['range']
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        if ans.sum() == 0:
            path['sample_id'] = []
            path['distribution'] = [0, 0]
            path['output'] = 0
            path['coverage'] = 0
        else:
            idx = np.flatnonzero(ans)
            path['sample_id'] = idx.tolist()
            pos = (ans == 1).sum()
            neg = (ans == -1).sum()
            path['distribution'] = [neg, pos]
            path['output'] = int(np.argmax(path['distribution']))
            path['coverage'] = 1.0 * len(idx) / X.shape[0]
        path['n_classes'] = 2

def assign_samples_lgbm(paths, data, model):
    X, y = data
    tree_paths = {}
    for p in paths:
        if p['tree_index'] not in tree_paths:
            tree_paths[p['tree_index']] = []
        tree_paths[p['tree_index']].append(p)

    if len(model.classes_) > 2:
        for tree_index in tree_paths:
            tpaths = tree_paths[tree_index]
            min_value = np.array([p['value'] for p in tpaths]).min()
            if min_value < 0:
                for p in tpaths:
                    p['value'] -= min_value

    weight = np.array([np.sum(y == c) for c in model.classes_])
    weight = weight.astype(np.float64)
    weight = 1.0 / (weight / weight.max())
    table = Table()
    for i in range(X.shape[1]):
        table.add(X[:, i])
    for i, path in enumerate(paths):
        m = path.get('range')
        missing = path.get('missing', [])
        conds = [(key, m[key], key in missing) for key in m]
        idx = table.query(conds)
        path['sample_id'] = idx
        ans = y[idx]
        path['distribution'] = [(ans == c).sum() for c in model.classes_]
        distri = np.array(path['distribution']) * weight
        if len(model.classes_) == 2:
            path['output'] = 0 if path['value'] < 0 else 1
            path['output_class'] = path['output']
            path['is_multiclass'] = False
        else:
            path['output'] = [0 for c in model.classes_]
            o = path['tree_index'] % len(model.classes_)
            path['output'][o] = path['value']
            path['is_multiclass'] = True
            path['output_class'] = o
        s = np.sum(distri)
        path['confidence'] = distri[path['output_class']] / s if s > 0 else 0
        path['classes'] = model.classes_
        path['n_classes'] = len(model.classes_)
        path['coverage'] = 1.0 * len(idx) / X.shape[0]

def assign_samples_RF(paths, data, model):
    X, y = data
    y = np.array(y)
    table = Table()
    for i in range(X.shape[1]):
        table.add(X[:, i])

    weight = np.array([np.sum(y == c) for c in model.classes_])
    weight = weight.astype(np.float64)
    weight = 1.0 / (weight / weight.max())
    # print('weight', weight)
    for i, path in enumerate(paths):
        m = path.get('range')
        missing = path.get('missing', [])
        conds = [(key, m[key], key in missing) for key in m]
        idx = table.query(conds)
        path['sample_id'] = idx
        ans = y[idx]
        path['distribution'] = [(ans == c).sum() for c in model.classes_]
        distri = np.array(path['distribution']) * weight
        o = np.argmax(distri)
        if path['distribution'][o] == 0:
            v = 0
            path['confidence'] = 0
        else:
            s = np.sum(distri)
            path['confidence'] = distri[o] / s
            v = (distri[o] / s) - 1.0 / len(distri)
            v = v ** 2
        if len(model.classes_) == 2:
            path['output'] = o
            path['output_class'] = o
            path['value'] = v * (-1 if o == 0 else 1)
            path['is_multiclass'] = False
        else:
            path['output'] = [0 for c in model.classes_]
            path['value'] = v
            path['output'][o] = path['value']
            path['is_multiclass'] = True
            path['output_class'] = o
        path['classes'] = model.classes_
        path['n_classes'] = len(model.classes_)
        path['coverage'] = 1.0 * len(idx) / X.shape[0]
        '''
        ans = 2 * y - 1
        m = path['range']
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        pos = (ans == 1).sum()
        neg = (ans == -1).sum()
        if pos + neg > 0:
            v = pos / (pos + neg) - 0.5
            path['value'] = (v ** 2) * (-1 if v < 0 else 1)
            #1 if pos > neg else -1
            if pos + neg > 0:
                path['confidence'] = max(pos, neg) / (pos + neg)
            else:
                path['confidence'] = 0
        else:
            path['value'] = 0
            path['confidence'] = 0
        '''

def path_extractor(model, model_type, data = None):
    if model_type == 'random forest' :
        ret = []
        for tree_index, estimator in enumerate(model.estimators_):
            treepaths = path_extractor(estimator, 'decision tree')
            #print('tree', tree_index, len(treepaths))
            for rule_index, path in enumerate(treepaths):
                path['tree_index'] = tree_index
                path['rule_index'] = rule_index
                path['name'] = 'r' + str(tree_index) + '_' + str(rule_index)
            ret += treepaths
        assign_samples_RF(ret, data, model)
        #if len(ret) > 30000:
        #    ret = sorted(ret, key = lambda x: -x['confidence'])
        #    ret = ret[:30000]
        #if len(ret) > 15000:
        #    ret = random.sample(ret, 15000)
        return ret
    elif model_type == 'node harvest':
        ret = []
        for tree_index, estimator in enumerate(model.estimators_):
            treepaths = path_extractor(estimator, 'node harvest tree')
            #print('tree', tree_index, len(treepaths))
            for rule_index, path in enumerate(treepaths):
                path['tree_index'] = tree_index
                path['rule_index'] = rule_index
                path['name'] = 'r' + str(tree_index) + '_' + str(rule_index)
            ret += treepaths
        model.classes_ = [0, 1]
        assign_samples_RF(ret, data, model)
        #if len(ret) > 30000:
        #    ret = sorted(ret, key = lambda x: -x['confidence'])
        #    ret = ret[:30000]
        #if len(ret) > 15000:
        #    ret = random.sample(ret, 15000)
        return ret
    elif model_type == 'lightgbm':
        ret = []
        info = model._Booster.dump_model()
        for tree_index, tree in enumerate(info['tree_info']):
            treepaths = visit_boosting_tree(tree['tree_structure'])
            for rule_index, path in enumerate(treepaths):
                path['tree_index'] = tree_index
                path['rule_index'] = rule_index
                path['name'] = 'r' + str(tree_index) + '_' + str(rule_index)
            ret += treepaths
        return ret
    elif model_type == 'decision tree':
        return visit_decision_tree(model.tree_)
    elif model_type == 'node harvest tree':
        return visit_decision_tree(model)
    return []

