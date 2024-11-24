import bisect
import math
import numpy as np
import pandas as pd
import copy
from scalable.model.utils import get_model
from scalable.anomaly import LRAnomalyDetection

def rule_to_text(rule):
    conds = rule[0]
    ret = rule[1]
    return 'IF ' + ' AND '.join(['%s %s %s' % (str(a), str(b), str(c)) for (a, b, c) in conds]) + ' THEN ' + ret

class ModelUtil():
    def __init__(self, data_name, model_name, parameters = None):
        self.data_name = data_name
        model = get_model(data_name, model_name)
        model.init_data()

        if parameters is not None:
            model.parameters = parameters
        model.train()
        # model.get_performance()
        data_table = model.data_table.copy()
        self.features = copy.deepcopy(model.features)
        while len(self.features) < model.X.shape[1]:
            k = str(len(self.features))
            data_table[k] = model.X[:, len(self.features)]
            self.features.append(k)
        self.min_coverage = 0.01

        new_feature = {}
        feature_pos = {}
        for index, feature in enumerate(self.features):
            is_cat = False
            if model.has_categorical_feature:
                for delimiter in [' - ', '_']:
                    if delimiter not in feature:
                        continue
                    # print('feature', feature)
                    name, _ = feature.split(delimiter)
                    if len([k for k in self.features if name in k]) == 1:
                        continue
                    if name not in new_feature:
                        new_feature[name] = {}
                    if feature not in new_feature[name]:
                        new_feature[name][feature] = index
                    is_cat = True
                    break
            if not is_cat:
                new_feature[feature] = index
        # print('features', self.features)
        feature_range = {}
        for key in new_feature:
            if key in data_table.columns:
                x = data_table[key]
                x = x[~np.isnan(x)]
                feature_range[key] = [x.min(), x.max()]
            elif type(new_feature[key]) != int:
                feature_range[key] = [0, len(new_feature[key])]
            if type(new_feature[key]) == int:
                index = new_feature[key]
                feature_pos[index] = (key, index)
            else:
                for feature in new_feature[key]:
                    index = new_feature[key][feature]
                    feature_pos[index] = (key, index)

        model.generate_path()
        self.model = model
        self.feature_range = feature_range
        self.feature_pos = feature_pos
        self.data_table = data_table

    def init_suffix_sum(self, X):
        self.ordered_vals = []
        for i in range(X.shape[1]):
            x = X[:, i]
            x = x[~pd.isnull(x)]
            vals = sorted(x.tolist())
            self.ordered_vals.append(vals)

    def get_sum(self, key, left, right, dl = 0, dr = 0):
        key = int(key)
        try:
            left = bisect.bisect_left(self.ordered_vals[key], left - 1e-6) - 1 + dl
            right = bisect.bisect_right(self.ordered_vals[key], right) + dr
            left = max(0, left)
            right = min(len(self.ordered_vals[key]), right)
            return right - left
        except:
            pass
        return 0
    
    def anomaly_score(self, paths):
        X_test, y_test = self.get_rule_matrix(paths)
        y_test = y_test.astype(int)
        score = self.anomaly.score(X_test, y_test)
        return score

    def init_anomaly_detection(self):
        X, y = self.get_rule_matrix()
        y = y.astype(int)
        idx = [i for i, p in enumerate(self.model.paths) if p['coverage'] > self.min_coverage]
        self.anomaly = LRAnomalyDetection(X[idx], y[idx])

    def interpret_path(self, path, to_text = False):
        conds = {}
        current_encoding = self.model.current_encoding
        data_table = self.data_table
        for k in path['range']:
            name = self.feature_pos[k][0]
            val = path['range'][k]
            if name in current_encoding:
                if name not in conds:
                    conds[name] = [1] * len(current_encoding[name])
                if name in data_table.columns:
                    for i in range(self.feature_range[name][0], self.feature_range[name][1]):
                        if i < val[0] or i > val[1]:
                            conds[name][i - self.feature_range[name][0]] = 0
                else:
                    if val[0] > 0:
                        conds[name] = [0] * len(current_encoding[name])
                        conds[name][self.feature_pos[k][1]] = 1
                    else:
                        conds[name][self.feature_pos[k][1]] = 0
            else:
                cond = [max(self.feature_range[name][0], val[0]), min(self.feature_range[name][1], val[1])]
                conds[name] = cond

        output_conds = []
        for name in conds:
            val = conds[name]
            op = 'is'
            value = ''
            if name in current_encoding:
                is_negation = np.sum(val) * 2 >= len(val) and len(val) > 2
                if is_negation:
                    op = 'is not'
                    for i, d in enumerate(val):
                        if d == 0:
                            value = value + ' and ' + current_encoding[name][i]
                    value = value[5:]
                else:
                    for i, d in enumerate(val):
                        if d == 1:
                            value = value + ' or ' + current_encoding[name][i]
                    value = value[4:]
            else:
                if val[0] == self.feature_range[name][0]:
                    op = '<='
                    value = (val[1])
                elif val[1] == self.feature_range[name][1]:
                    op = '>='
                    value = (val[0])
                else:
                    op = 'in'
                    value = '%.2f to %.2f' % ((val[0]), (val[1]))
            output_conds.append((name, op, value))
        try:
            output_label = self.model.output_labels[path['output']]
        except:
            output_label = self.model.output_labels[np.argmax(path['output'])]
        if to_text:
            return rule_to_text((output_conds, output_label))
        else:
            return output_conds, output_label

    def check_path(self, path, X, byclass = False):
        n_samples = len(X)

        m = path['range']
        cover = np.ones(n_samples)
        for key in m:
            name = self.feature_pos[key][0]
            if name in self.model.categorical_data:
                cover = cover * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
            else:
                cond = [int(max(self.feature_range[name][0], m[key][0])), int(min(self.feature_range[name][1], m[key][1]))]
                center_count = self.get_sum(int(key), cond[0], cond[1])
                remain_count = n_samples - center_count
                for i in range(len(cover)):
                    dist = 0
                    val = X[i, int(key)]
                    if val < cond[0]:
                        dist = self.get_sum(int(key), val, cond[0], 0, -1) / remain_count
                    elif val > cond[1]:
                        dist = self.get_sum(int(key), cond[1], val, 1, 0) / remain_count
                    cover[i] *= (1 - dist)
        if byclass:
            conf = np.array(path['distribution']).astype('float') ** 2
            if conf.sum() >= 1:
                conf /= conf.sum()
            cover = (cover.repeat(2).reshape((n_samples, 2)) * conf).reshape(-1)
        return cover

    def get_rule_matrix(self, paths = None):
        feature_pos = copy.deepcopy(self.feature_pos)

        is_feature_categorical = {}
        feature_val_idxs = {}
        for i in self.feature_pos:
            if i == -1:
                continue
            feature, _ = self.feature_pos[i]
            if feature in is_feature_categorical:
                is_feature_categorical[feature] = True
            else:
                is_feature_categorical[feature] = False

        for i in self.feature_pos:
            if i == -1:
                continue
            feature, _ = self.feature_pos[i]
            if is_feature_categorical[feature]:
                if feature not in feature_val_idxs:
                    feature_val_idxs[feature] = []
                if i >= 0:
                    feature_val_idxs[feature].append(i)

        is_categorical = []
        feature_range = []
        for i in self.feature_pos:
            if i == -1:
                continue
            feature, index = self.feature_pos[i]
            is_categorical.append(is_feature_categorical[feature])
            feature_range.append(self.feature_range.get(feature, [0, 1]))

        if paths is None:
            paths = self.model.paths
        n_features = len(is_categorical)

        for i in self.model.to_category_idx:
            name = self.feature_pos[i][0]
            is_categorical[i] = True
            feature_len =  self.feature_range[name][1] - self.feature_range[name][0]
            idx = [i] + [j for j in range(n_features, n_features + feature_len - 1)]
            for it, j in enumerate(idx):
                feature_pos[j] = (name, it)
                is_categorical.append(True)
            feature_val_idxs[name] = idx
            n_features += feature_len - 1

        n_extends = 0
        if self.model.model_name == 'lightgbm' or self.model.model_name == 'lgbm':
            n_extends = n_features
        X = np.ones((len(paths), n_features + n_extends)).astype('float') * 0.5
        y = np.ones((len(paths), ))

        mid = np.zeros((n_features + n_extends)).astype('float')
        data_table = self.data_table
        self.init_suffix_sum(self.model.X)

        valid_counts = np.zeros(n_features)
        for i, k in enumerate(self.features):
            valid_counts[i] = len(data_table) - data_table[k].isna().sum()
        n_samples = len(data_table)
        for i, feature in enumerate(self.features):
            if is_categorical[i]:
                if i in self.model.to_category_idx:
                    vmin = data_table[feature].min()
                    vmax = data_table[feature].max()
                    for k, j in enumerate(feature_val_idxs[feature]):
                        val = (data_table[feature] == k + vmin).sum() / n_samples
                        mid[j] = val
                else:
                    mid[i] = (data_table[feature] == 1).sum() / n_samples

        for row_i, p in enumerate(paths):
            m = p['range']
            row = mid.copy()
            for i in p.get('missing', []):
                row[i + n_features] = 1
            for i in m:
                if is_categorical[i]:
                    feature, _ = self.feature_pos[i]
                    idx = feature_val_idxs[feature]
                    if i in self.model.to_category_idx:
                        ll = feature_range[i][0]
                        rr = feature_range[i][1]
                        left = int(max(ll, math.floor(m[i][0])))
                        right = int(min(rr, math.ceil(m[i][1])))
                        if ll > 0:
                            left -= ll
                            right -= ll
                            rr -= ll
                            ll = 0
                        rr = min(rr, len(idx))
                        right = min(right, rr)
                        for j in range(rr):
                            row[idx[j]] = 0
                        for j in range(left, right):
                            row[idx[j]] = mid[idx[j]]
                    else:
                        if m[i][1] > 1:
                            if row[idx].sum() == 1:
                                row[idx] = 0
                            row[i] = mid[i]
                        else:
                            row[i] = 0
                else:
                    left = (max(feature_range[i][0], m[i][0]))
                    right = (min(feature_range[i][1], m[i][1]))
                    left = self.get_sum(i, 0, left - 1e-6, 1, 0) / len(self.model.X)
                    right = self.get_sum(i, 0, right, 1, 0) / len(self.model.X)
                    print(row_i, self.feature_pos[i][0], left, right)
                    #left = self.get_sum(i, 0, left - 1e-6, 1, 0) / valid_counts[i]
                    #right = self.get_sum(i, 0, right, 1, 0) / valid_counts[i]
                    row[i] = (left + right) / 2 - 0.5
            for feature in feature_val_idxs:
                idx = feature_val_idxs[feature]
                tot = row[idx].sum()
                if tot > 0:
                    row[idx] /= tot

            X[row_i] = row
            if type(p['output']) == list:
                y[row_i] = np.argmax(p['output'])
            else:
                y[row_i] = p['output']

        self.feature_val_idxs = feature_val_idxs
        current_encoding = self.model.current_encoding
        self.feature_name = []
        for i in range(n_features):
            name, k = feature_pos[i]
            if name in current_encoding and is_categorical[i]:
                name = name + ' ' + current_encoding[name][k]
            self.feature_name.append(name)

        y = y.astype(int)
        return X, y

    def get_cover_matrix(self, X, normalize = False, fuzzy = False, byclass = False):
        paths = self.model.paths
        if not fuzzy:
            mat = np.array([p['sample'] for p in paths]).astype('float')
        else:
            if byclass:
                mat = np.array([self.check_path(p, X, byclass=True) for p in paths]).astype('float')
            else:
                mat = np.array([self.check_path(p, X) for p in paths]).astype('float')

        if normalize:
            for i, path in enumerate(paths):
                sum = np.sqrt(np.sum(mat[i]))
                if sum > 0:
                    mat[i] /= sum
        return mat

def export_rules_to_csv(filename, model, idxes):
    rules = []
    max_n_conds = 0
    for it, i in enumerate(idxes):
        conds, output = model.interpret_path(model.paths[i])
        rules.append({'cond': conds, 'predict': output, 'index': i, 'order': it, 'attr': 0 })
        max_n_conds = max(len(conds), max_n_conds)
    conds_per_line = 4
    max_n_conds = math.ceil(max_n_conds / conds_per_line) * conds_per_line

    f = open(filename + '.csv', 'w')

    for it, rule in enumerate(rules):
        s = '' + str(rule['order'])
        line = 0
        n_conds = len(rule['cond'])
        n_lines = math.ceil(n_conds / conds_per_line)
        index = rule['index']

        for line in range(n_lines):
            if line == 0:
                s += ',#%d,IF,' % (index)
            else:
                s += ',,,'
            for pos in range(conds_per_line):
                i = pos + line * conds_per_line
                if i < n_conds:
                    item = rule['cond'][i]
                    s += item[0] + ',' + item[1] + ',' + str(item[2]) + ','
                    s += 'AND,' if i < n_conds - 1 else '...,'
                else:
                    s += '...,...,...,...,'
            if line == n_lines - 1:
                s = s[:-4]
                s += 'THEN,%s,%d,%3f' % (rule['predict'], np.sum(model.paths[index]['distribution']), model.paths[index]['confidence'])
            s += '\n'
        f.write(s + '\n')
    f.close()
