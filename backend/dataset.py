
from random import *
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from annoy import AnnoyIndex
from dataconfig import cache_dir_path, data_encoding, data_setting
import sys
sys.path.append('..')
from scalable.model_utils import ModelUtil
import os

class DataLoader():
    def __init__(self, info, name, target):
        self.info = info
        self.name = name

        utils = ModelUtil(info['model_info']['dataset'], info['model_info']['model'].lower())
        self.utils = utils
        model = utils.model
        self.model = model
        self.paths = self.info['paths']
        # print('feature', self.info['features'])
        conds, vectors, defaults = utils.get_feature_representation(self.info['features'], self.paths)
        vectors = [p['feature_vector'] for p in self.paths]

        # scores = np.array([p['score'] for p in self.paths])
        self.path_index = {}
        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
            path['level'] = 0 if path['represent'] else 1
        max_level = 2
        self.selected_indexes = [path['name'] for path in self.paths if path['represent']]#self.model['selected']
        self.features = self.info['features']
        for i, feature in enumerate(self.features):
            feature['default'] = defaults[i]

        _name = name
        if 'stock' in _name:
            _name = 'stock'
        elif 'credit' in _name:
            _name = 'credit'
        self._name = _name
        current_encoding = data_encoding.get(_name, {})
        current_setting = data_setting.get(_name, {})
        self.current_encoding = current_encoding
        self.current_setting = current_setting
        
        self.info['model_info']['target'] = target
        self.target = target
        if self.info['model_info']['model'] == 'LightGBM':
            self.info['model_info']['weighted'] = True
        else:
            self.info['model_info']['weighted'] = False

        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path)

        distribs = []
        for i, path in enumerate(self.paths):
            distrib = np.array(path['distribution'])
            distrib = distrib / distrib.sum()# * 0.5
            distribs.append(distrib)
        max_sum = np.sqrt(np.max([(vector * vector).sum() for vector in vectors]))
        for i in range(len(vectors)):
            vectors[i] /= max_sum

        mats = []
        max_sum = np.sqrt(np.max([(vector * vector).sum() for vector in vectors]))
        for i in range(len(vectors)):
            output = np.zeros(len(distribs[i]))
            output[np.argmax(distribs[i])] = 1
            feature = np.concatenate((vectors[i], output), axis = 0)
            mats.append(feature)
            self.paths[i]['feature'] = feature.tolist()
            self.paths[i]['cond_norm'] = conds[i]

        path_mat = np.array(mats)
        np.seterr(divide='ignore', invalid='ignore')
        path_mat = path_mat.astype(np.float32)
        path_dist = pairwise_distances(X = path_mat, metric='euclidean')
        tree = AnnoyIndex(len(path_mat[0]), 'euclidean')
        for i in range(len(path_mat)):
            tree.add_item(i, path_mat[i])
        tree.build(10)
        self.tree = tree

        for i in range(len(self.paths)):
            p = self.paths[i]
            p['anomaly'] = p['initial_anomaly'] = p['score']
            p['represent'] = False
            p['children'] = []

        self.path_dist = path_dist
        for level in range(0, max_level):
            ids = []
            for i in range(len(self.paths)):
                if self.paths[i]['level'] == level:
                    ids.append(i)
            for i in range(len(self.paths)):
                if self.paths[i]['level'] == level + 1:
                    self.paths[i]['children'] = []
                    nearest = -1
                    nearest_dist = 1e10
                    for j in ids:
                        if path_dist[i][j] < nearest_dist:# and self.paths[i]['output'] == self.paths[j]['output']:
                            nearest = j
                            nearest_dist = path_dist[i][j]
                    j = nearest
                    # self.paths[i]['father'] = j
                    self.paths[j]['children'].append(i)
                    
        for i in range(len(self.paths)):
            self.paths[i]['father'] = i
        for i in range(len(self.paths)):
            children = [(path_dist[i][j], j) for j in self.paths[i]['children']]
            children = sorted(children)
            p = self.paths[i]
            #p['children'] = [j for _, j in children]
            #p['children_dist'] = [d for d, j in children]
            p['children_'] = [j for _, j in children]
            p['children_dist_'] = [d for d, j in children]
            for j in p['children_']:
                self.paths[j]['father'] = i

        self.path_dict = {}
        for path in self.paths:
            self.path_dict[path['name']] = path

    def get_general_info(self, idxes = None):
        if idxes is None:
            positives = (self.data_table['Label'] == self.target_class).sum()
            total = len(self.data_table['Label'])
        else:
            idxes = [i for i in idxes if i < len(self.data_table)]
            positives = (self.data_table['Label'][idxes] == self.target_class).sum()
            total = len(idxes)
        return (positives, total, positives / total)

    def get_relevant_samples(self, idxes):
        samples = {}
        for i in idxes:
            for j in self.paths[i]['sample_id']:
                if j not in samples:
                    samples[j] = 1
                else:
                    samples[j] += 1
        thres = 1
        if len(samples) * 2 > len(self.model.data_table):
            thres += 1
        ret = [k for k in samples if samples[k] >= thres]
        ret = sorted(ret)
        return ret
    
    def init_hierarchy(self):
        for i in range(len(self.paths)):
            self.paths[i]['father'] = i
        for i in range(len(self.paths)):
            p = self.paths[i]
            p['children'] = p['children_']
            p['children_dist'] = p['children_dist_']
            for j in p['children_']:
                self.paths[j]['father'] = i
        for i in range(len(self.paths)):
            p = self.paths[i]
            if p['father'] == i:
                p['level'] = 0
            else:
                p['level'] = 1
    
    def adjust_hierarchy(self, fathers, sons):
        path_dist = self.path_dist
        fathers_ = set(fathers)
        sons = [i for i in sons if i not in fathers_]
        for i in sons:
            self.paths[i]['children'] = []
            self.paths[i]['children_dist'] = []
            nearest = -1
            nearest_dist = 1e10
            for j in fathers:
                if path_dist[i][j] < nearest_dist:
                    nearest = j
                    nearest_dist = path_dist[i][j]
            j = nearest
            self.paths[j]['children'].append(i)
            self.paths[j]['children_dist'].append(path_dist[i][j])
            self.paths[i]['father'] = j
            self.paths[i]['level'] = self.paths[j]['level'] + 1
        for i in fathers:
            children = [(path_dist[i][j], j) for j in self.paths[i]['children']]
            children = sorted(children)
            p = self.paths[i]
            p['children'] = [j for _, j in children]
            p['children_dist'] = [d for d, j in children]
    
    def is_zoomin(self, idxes):
        flag = True
        for i in idxes:
            if i not in self.current_idxes:
                flag = False
                break
        return flag
    
    def set_current_rules(self, idxes):
        self.current_idxes = idxes

    def get_encoded_path(self, idx):
        path = self.paths[idx]
        output = path['output']
        if type(output) != int:
            output = np.argmax(output)
        # print('self.class_weight', self.class_weight)
        distribution = np.array(path['distribution']) * self.class_weight
        confidence = distribution[output] / distribution.sum()
        
        return {
            'labeled': False,
            'name': path['name'],
            'idx': idx,
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'father': path['father'],
            'range': path['range'],
            'cond_norm': path['cond_norm'],
            'missing': path.get('missing', []),
            'level': path['level'],
            'weight': path['weight'],
            'anomaly': path['anomaly'],
            'initial_anomaly': path['initial_anomaly'],
            'num_children': len(path['children']),
            'distribution': path['distribution'],
            'confidence': confidence,
            'coverage': path['coverage'],
            'feature_vector': path['feature'],
            'output': output,
            'samples': path['sample_id'],
        }

    def model_info(self):
        return self.info['model_info']

    def set_data_table(self, data, classes = None):
        # print('data length', len(self.model.y), len(data))
        if len(self.model.y_train) == len(data):
            pred_y = self.model.clf.predict(self.model.X_train)
        else:
            if self._name == 'credit':
                data['PriorDefault'] = 1 - data['PriorDefault']
            X, _ = self.model.transform_data(data)
            pred_y = self.model.clf.predict(X)
            if self._name == 'credit':
                data['PriorDefault'] = 1 - data['PriorDefault']


        data['Predict'] = pred_y
        col = data[self.target].values
        data = data.drop(self.target, axis = 1)
        data['Label'] = col

        setting = self.current_setting
        encoding = self.current_encoding

        for index, feature in enumerate(self.features):
            name = feature['name']
            if name in setting and 'scale' in setting[name]:
                feature['scale'] = setting[name]['scale']
            else:
                feature['scale'] = 'linear'

            if name in setting and 'display_name' in setting[name]:
                feature['display_name'] = setting[name]['display_name']

            if name in encoding:
                feature['dtype'] = 'category'
                if data[name].dtype == np.int64:
                    if len(encoding[name]) > 0:
                        feature['values'] = encoding[name]
                        col = data[name].values
                        data = data.drop(name, axis = 1)
                        data[name] = [encoding[name][i] for i in col]
                    else:
                        feature['values'] = np.unique(data[name].values).tolist()
                        feature['values'] = [str(k) for k in feature['values']]
                else:
                    if len(encoding[name]) > 0:
                        feature['values'] = encoding[name]
                    else:
                        feature['values'] = np.unique(data[name].values).tolist()
                        feature['values'] = [str(k) for k in feature['values']]
                #print(name, feature['values'])
            else:
                try:
                    r = feature['values'] = feature['range'] = data[name].astype(float).quantile([0.01, 0.99]).tolist()
                    if r[1] - r[0] > 100:
                        r = feature['values'] = feature['range'] = data[name].astype(float).quantile([0.05, 0.95]).tolist()
                    q = data[name].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
                    data.loc[data[name] < r[0], name] = r[0]
                    data.loc[data[name] > r[1], name] = r[1]
                    feature['avg'] = data[name].mean()
                    feature['q'] = q
                except:
                    pass

            if name not in encoding and data[name].dtype != 'object':
                r = feature['range']
                data.loc[data[name] < r[0], name] = r[0]
                data.loc[data[name] > r[1], name] = r[1]

        if classes is None:
            classes = self.paths[0]['classes']
        else:
            for p in self.paths:
                #p['output_class'] = p['output']
                #p['output'] = classes[p['output']]
                p['classes'] = classes
            try:
                data['Predict'] = [classes[i] for i in pred_y]
            except:
                data['Predict'] = [i for i in pred_y]
            # print('Label', data['Label'].dtype)
            if data['Label'].dtype == 'int64':
                data['Label'] = [classes[i] for i in data['Label']]
        targets = [(i, j) for i, j in enumerate(classes)]
        self.info['model_info']['targets'] = [x[1] for x in targets]
        weight = np.array([(data['Label'] == c).sum() for c in classes]).astype(np.float64)
        if weight.max() == 0:
            classes = self.model.output_labels
            weight = np.array([(data['Label'] == c).sum() for c in classes]).astype(np.float64)
        #print("data['Label']", data['Label'], classes)
        weight = 1.0 / (np.maximum(np.ones(len(weight)), weight) / weight.max())
        self.class_weight = weight
        self.target_class = classes[1]

        data = data.fillna(-1)
        self.data_table = data

class DatasetLoader():
    def __init__(self):
        data_loader = {}

        data_table = pd.read_csv('../data/case1_credit_card/step0.csv')
        info = pickle.load(open('../output/case/credit0.pkl', 'rb'))
        loader = DataLoader(info, 'credit', 'Approved')
        loader.set_data_table(data_table, classes = ['Rejected', 'Approved'])
        data_loader['credit0'] = loader
        
        data_table = pd.read_csv('../data/case1_credit_card/step1.csv')
        info = pickle.load(open('../output/case/credit1.pkl', 'rb'))
        loader = DataLoader(info, 'credit', 'Approved')
        loader.set_data_table(data_table, classes = ['Rejected', 'Approved'])
        data_loader['credit1'] = loader

        data_table = pd.read_csv('../data/case1_credit_card/step1.csv')
        info = pickle.load(open('../output/case/credit2.pkl', 'rb'))
        loader = DataLoader(info, 'credit', 'Approved')
        loader.set_data_table(data_table, classes = ['Rejected', 'Approved'])
        data_loader['credit2'] = loader
        '''
        data_table = pd.read_csv('../data/obesity.csv')
        info = pickle.load(open('../output/case/obesity.pkl', 'rb'))
        loader = DataLoader(info, 'obesity', 'NObeyesdad')
        loader.set_data_table(data_table)
        data_loader['obesity'] = loader
        
        data_table = pd.read_csv('../data/obesity.csv')
        info = pickle.load(open('../output/case/obesity2.pkl', 'rb'))
        loader = DataLoader(info, 'obesity', 'NObeyesdad')
        loader.set_data_table(data_table)
        data_loader['obesity1'] = loader
        
        data_table = pd.read_csv('../data/drybean.csv')
        info = pickle.load(open('../output/case/drybean.pkl', 'rb'))
        loader = DataLoader(info, 'drybean', 'Class')
        loader.set_data_table(data_table)#, classes = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMOSAN', 'HOROZ', 'SIRA'])
        data_loader['drybean'] = loader
        
        data_table = pd.read_csv('../data/drybean.csv')
        info = pickle.load(open('../output/case/drybean2.pkl', 'rb'))
        loader = DataLoader(info, 'drybean', 'Class')
        loader.set_data_table(data_table)#, classes = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMOSAN', 'HOROZ', 'SIRA'])
        data_loader['drybean1'] = loader
        
        data_table = pd.read_csv('../data/case2_stock/step/3year_raw_5.csv')
        info = pickle.load(open('../output/case/stock_step3.pkl', 'rb'))
        loader = DataLoader(info, 'stock', 'label')
        loader.set_data_table(data_table, classes = ["decrease", "increase", "stable"])
        data_loader['stock2'] = loader

        data_table = pd.read_csv('../data/case2_stock/step/3year_raw_3.csv')
        info = pickle.load(open('../output/case/stock_step1.pkl', 'rb'))
        loader = DataLoader(info, 'stock', 'label')
        loader.set_data_table(data_table, classes = ["decrease", "increase", "stable"])
        data_loader['stock1'] = loader
        
        data_table = pd.read_csv('../data/case2_stock/step/3year_raw_3.csv')
        info = pickle.load(open('../output/case/stock_step0.pkl', 'rb'))
        loader = DataLoader(info, 'stock', 'label')
        loader.set_data_table(data_table, classes = ["decrease", "increase", "stable"])
        data_loader['stock'] = loader
        
        '''
        #loader.discretize()

        self.data_loader = data_loader

    def get(self, name):
        return self.data_loader[name]

