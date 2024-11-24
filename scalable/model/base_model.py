import sys
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from scalable.algorithm.tree_extractor import path_extractor, assign_samples_RF, assign_samples_lgbm
from scalable.config import cache_path

class BaseModel():
    def __init__(self):
        self.data_name = ''
        self.model_name = ''
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None
        self.data_table = None
        self.clf = lambda x: x
        self.test_size = 0.25
        self.parameters = {}
        self.feature_values = {}
        self.transform = {}
        self.current_encoding = {}
        self.categorical_data = []
        self.to_category_idx = []
        self.model_id = 0
        self.output_labels = [0, 1]
        self.has_categorical_feature = False

    def init_data(self):
        pass

    def check_columns(self, data_table, target):
        self.features = [feature for feature in data_table.columns if feature != target]
        #self.features = sorted(self.features)

    def train(self):
        if self.model_name == 'random forest' or self.model_name == 'rf':
            self.clf = RandomForestClassifier(**self.parameters)
        elif self.model_name == 'lightgbm' or self.model_name == 'lgbm':
            self.clf = LGBMClassifier(**self.parameters)
        self.clf.fit(self.X_train, self.y_train)

    def transform_data(self, data):
        return data.drop(self.target, axis = 1), None

    @property
    def generated_path_cache(self):
        return os.path.join(cache_path, '%s_%s_%d_raw_paths.pkl' % (self.data_name, self.model_name, self.model_id))

    def generate_path(self):
        if self.model_id != -1 and os.path.exists(self.generated_path_cache):
            self.paths = pickle.load(open(self.generated_path_cache, 'rb'))
            print('%d rules loaded.' % len(self.paths))
        else:
            if self.model_name == 'random forest' or self.model_name == 'rf':
                self.paths = path_extractor(self.clf, 'random forest', (self.X_train, self.y_train))
            elif self.model_name == 'lightgbm' or self.model_name == 'lgbm':
                self.paths = path_extractor(self.clf, 'lightgbm')
                assign_samples_lgbm(self.paths, (self.X, self.y), self.clf)
            self.paths = [path for path in self.paths if len(path['sample_id']) > 0]
            for index, path in enumerate(self.paths):
                path['index'] = index
            if self.model_id != -1:
                pickle.dump(self.paths, open(self.generated_path_cache, 'wb'))
            print('%d rules generated.' % len(self.paths))
    @property
    def n_classes(self):
        return len(self.output_labels)

    def set_paramters(self, parameters):
        self.parameters = parameters

    def get_performance(self, X_train = None, y_train = None, X_test = None, y_test = None):
        if X_train is None:
            X_test = self.X_test
            X_train = self.X_train
            y_test = self.y_test
            y_train = self.y_train

        num_classes = len(self.output_labels)

        if len(self.output_labels) == 2:
            y_pred = self.clf.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            prec = precision_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)
            # print('TRAIN')
            print(f'Accuracy Score:  {round(acc, 4)}')
            # print(f'Precision Score:  {round(prec, 4)}')
            # print(f'F1 Score:  {round(f1, 4)}')

            y_pred = self.clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf = self.clf.predict_proba(X_test)
            margin = [abs(conf[i][1] - conf[i][0]) for i, j in enumerate(y_test)]
            conf = [conf[i][j] for i, j in enumerate(y_test)]# if y_test[i] == y_pred[i]]
            conf = np.mean(conf)
            margin = np.mean(margin)

            y_pred_proba = self.clf.predict_proba(X_test)
            cross_entropy = log_loss(y_test, y_pred_proba)

            y_pred_proba = self.clf.predict_proba(X_train)
            cross_entropy_train = log_loss(y_train, y_pred_proba)

            # print('TEST')
            print(f'Accuracy Score:  {round(acc, 4)}')
            print(f'Precision Score:  {round(prec, 4)}')
            print(f'F1 Score:  {round(f1, 4)}')
            print(f'Confidence: {round(conf, 4)}')
            # print(f'Margin: {round(margin, 4)}')
            # print(f'Cross entropy: {round(cross_entropy, 4)}')

            return (acc, prec, f1)
        else:
            from sklearn.metrics import roc_auc_score, auc
            from sklearn.preprocessing import label_binarize

            classes = self.output_labels
            y_pred_prob = self.clf.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=classes)

            y_pred = self.clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # print('TEST')
            print(f'Accuracy Score:  {round(acc, 4)}')
            conf_mat = confusion_matrix(y_test, y_pred)

            try:
                auc_list = []
                for i in range(num_classes):
                    for j in range(i+1, num_classes):
                        y_i = y_pred_prob[:, i]
                        y_j = y_pred_prob[:, j]
                        true_i = y_test_bin[:, i]
                        true_j = y_test_bin[:, j]

                        y_ij = np.column_stack((y_i, y_j))
                        true_ij = np.column_stack((true_i, true_j))

                        # 计算AUC并添加到列表中
                        auc = roc_auc_score(true_ij, y_ij)
                        # print(f'AUC between {classes[i]} and {classes[j]}: ', auc)
                        auc_list.append(auc)
                # 计算平均AUC
                mean_auc = np.mean(auc_list)
                # print("Average AUC:", mean_auc)
            except:
                mean_auc = 0

            return (acc, mean_auc, 0)
