from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
from scalable.model.base_model import BaseModel
from scalable.anomaly import LRAnomalyDetection
from scalable.model.data_encoding import stock_encoding
import numpy as np
from scalable.config import data_path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score

random_state = 42

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/step/3year_3.csv')
        self.data_table = pd.read_csv(self.data_path)

        self.has_categorical_feature = True
        self.test_data_path = os.path.join(data_path, 'case2_stock/step/3month_3.csv')
        self.test_data_table = pd.read_csv(self.test_data_path)

        self.target = 'label'
        self.output_labels = ["decrease", "increase", "stable"]
        # self.model_id = 105

        self.model_id = -1
        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 200,
                'max_depth': 13,
                'max_leaf_nodes': 100,
                'class_weight': {0: 3.5, 1: 5.5, 2: 1, 'decrease': 3.5, 'increase': 5.5, 'stable': 1},
                #'class_weight': 'balanced',
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 500,
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0281,
                "subsample": 0.8789,
                "max_depth": 8,
                "num_leaves": 50,
                'class_weight': 'balanced',
                'verbosity': -1,
            }

    def init_data(self):
        self.data_table = self.data_table.drop('date', axis=1)
        self.test_data_table = self.test_data_table.drop('date', axis=1)
        if self.model_name == 'random forest' or self.model_name == 'rf':
            self.data_table = self.data_table.fillna(0)
            self.test_data_table = self.test_data_table.fillna(0)


        data_table = self.data_table.drop('ticker', axis=1)
        data_table = data_table.drop('newPrice', axis = 1)
        data_table = data_table.drop('currentPrice', axis = 1)
        features = data_table.columns.tolist()
        features = [k for k in features if k != 'rating' and k != 'label']
        print(f'{len(features)} features')

        for key in stock_encoding:
            index = 0
            for i in range(len(features)):
                if key in features[i]:
                    features[i] = key + '_' + stock_encoding[key][index]
                    index += 1

        X_train = data_table[features]
        y_train = data_table[self.target]
        X_test = self.test_data_table[features]
        y_test = self.test_data_table[self.target]

        self.train_rating = data_table['rating'].values
        self.test_rating = self.test_data_table['rating'].values
        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_test = y_test.values
        self.X = X_train.values
        self.y = y_train.values
        self.data_table = data_table.drop('rating', axis = 1)
        self.current_features = features

        self.check_columns(self.data_table, self.target)

if __name__ == '__main__':
    model = Model('random forest')
    model.init_data()
    # print(model.X_train.mean())
    model.train()
    model.get_performance()

    y_pred = model.clf.predict(model.X_train)
    conf_mat = confusion_matrix(model.y_train, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Train Accuracy on {model.output_labels[i]}: {accuracy}')

    y_pred = model.clf.predict(model.X_test)
    conf_mat = confusion_matrix(model.y_test, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Test Accuracy on {model.output_labels[i]}: {accuracy}')

    bank_idx = np.flatnonzero(model.test_data_table['industry_Banks—Regional'])
    X_test = model.X_test[bank_idx]
    y_test = model.y_test[bank_idx]
    y_pred = model.clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Bank Test Accuracy on {model.output_labels[i]}: {accuracy}')
    accuracy = accuracy_score(y_test, y_pred)
    print(round(accuracy, 4))
    '''
    bank_idx = np.flatnonzero(model.test_data_table['industry_Banks—Regional'])
    X_test = model.X_train[bank_idx]
    y_test = model.y_train[bank_idx]
    y_pred = model.clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Bank Train Accuracy on {model.output_labels[i]}: {accuracy}')
    accuracy = accuracy_score(y_test, y_pred)
    print(round(accuracy, 4))
    '''
    clf = model.clf
    sorted_features = [(clf.feature_name_[i], clf.feature_importances_[i]) for i in range(len(clf.feature_importances_))]
    sorted_features = sorted(sorted_features, key = lambda x: -x[1])
    for k in sorted_features:
        i, j = k
        i = model.current_features[int(i.split('_')[1])]
        # print(i, j)

    # model.generate_path()
    '''
    from scalable.model_utils import ModelUtil
    util = ModelUtil(data_name = 'stock_step0', model_name = 'lightgbm')
    X, y = util.get_rule_matrix()
    y = y.astype(int)
    detect = LRAnomalyDetection(X, y)
    score = detect.score()
    print('score', score.mean())
    np.save('score.npy', score)
    '''
