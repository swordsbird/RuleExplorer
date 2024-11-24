from sklearn.model_selection import train_test_split
import os

import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from scalable.config import data_path
from scalable.model.base_model import BaseModel
from scalable.anomaly import LRAnomalyDetection
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from scipy import stats

random_state = 42

new_label = {
    10: 0, 49: 0, 206: 0, 270: 0, 592: 0, 606: 0, 621: 0, 622: 0,
}
conds = ['cond2', 'cond3', 'cond4', 'cond5', 'cond6', 'cond7', 'cond8', 'cond9', 'cond10', 'cond12', 'cond13', 'cond14', 'cond16', 'cond17', 'cond18', 'cond19']

meta_json = {
    "columns": {
        "Gender": {
            "sdtype": "categorical"
        },
        "Age": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Debt": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "YearsEmployed": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "PriorDefault": {
            "sdtype": "categorical"
        },
        "Employed": {
            "sdtype": "categorical"
        },
        "DriversLicense": {
            "sdtype": "categorical"
        },
        "Citizen": {
            "sdtype": "categorical"
        },
        "ZipCode": {
            "sdtype": "categorical"
        },
        "Approved": {
            "sdtype": "categorical"
        },
        "Income": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "CreditScore": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Married": {
            "sdtype": "categorical"
        },
        "BankCustomer": {
            "sdtype": "categorical"
        },
        "Job": {
            "sdtype": "categorical"
        },
        "Ethnicity": {
            "sdtype": "categorical"
        }
    },
    "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1"
}


class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'credit'
        self.data_path = os.path.join(data_path, 'case1_credit_card/credit_card_train1.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.data_table_original = self.data_table.copy()
        self.target = 'Approved'
        self.model_id = 5
        self.output_labels = ['0', '1']

        self.has_categorical_feature = True
        self.test_data_path = os.path.join(data_path, 'case1_credit_card/test.csv')
        self.test_data_table = pd.read_csv(self.test_data_path)

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 13,
                'min_samples_leaf': 7,
                'max_features': 7,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 100,
                'max_depth': 12,
                'random_state': random_state,
            }

    def synthesize_data(self, label, num_rows):
        metadata = SingleTableMetadata.load_from_dict(meta_json)
        #synthesizer = SingleTablePreset(metadata, name='FAST_ML')
        synthesizer = CTGANSynthesizer(metadata)
        idx = self.data_table_original['Approved'] == label
        original_data = self.data_table_original[idx]
        synthesizer.fit(original_data)
        data = synthesizer.sample(num_rows=num_rows)
        data.loc['Approved', :] = label
        return data

    def transform_table(self, data_table):
        data_table = data_table.copy()
        data_table['ZipCode'] = [-1 if i in self.other_zipcodes else i for i in data_table['ZipCode'].values]
        for feature in self.qualitative_features:
            if feature not in self.binary_features:
                for val in self.feature_values[feature]:
                    data_table[feature + ' - '+ str(val)] = data_table[feature].values == val
        for feature in self.qualitative_features:
            if feature not in self.binary_features:
                data_table = data_table.drop(feature, axis = 1)
        return data_table

    def transform_data(self, data_table):
        data_table = self.transform_table(data_table)

        X = data_table[self.data_table.columns].drop(self.target, axis=1).values
        y = data_table[self.target].values
        return X, y

    def init_data_original(self, sampling_rate = 1):
        qualitative_features = ['Gender', 'Married', 'BankCustomer', 'Job', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen', 'ZipCode']
        self.qualitative_features = qualitative_features
        binary_features = []
        data_table = self.data_table

        data_table['Age'] = [np.floor(x) for x in data_table['Age'].values]

        unique_values = np.unique(data_table['ZipCode'].values)
        unique_values = sorted(unique_values)
        unique_groups = []
        for val in unique_values:
            unique_groups.append((val, (data_table['ZipCode'].values == val).sum()))
        other_zipcodes = [zc for zc, zc_count in unique_groups if zc_count < 30]
        new_zipcode = [-1 if i in other_zipcodes else i for i in data_table['ZipCode'].values]
        self.other_zipcodes = other_zipcodes
        data_table['ZipCode'] = new_zipcode
        unique_values = np.unique(new_zipcode)
        unique_values = sorted(unique_values)
        self.feature_values['ZipCode'] = unique_values
        self.transform['ZipCode'] = new_zipcode
        for feature in qualitative_features:
            unique_values = np.unique(data_table[feature].values)
            unique_values = sorted(unique_values)
            self.feature_values[feature] = unique_values
            if len(unique_values) == 2:
                unique_values = unique_values[1:]
                binary_features.append(feature)
            else:
                for i, val in enumerate(unique_values):
                    data_table[feature + ' - '+ str(val)] = data_table[feature].values == val
        for feature in qualitative_features:
            if feature not in binary_features:
                data_table = data_table.drop(feature, axis = 1)
        self.binary_features = binary_features

        features = data_table.drop(self.target, axis=1).columns
        X = data_table.drop(self.target, axis=1).values
        y = data_table[self.target].values
        indices = np.arange(X.shape[0])
        indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=self.test_size, random_state=random_state)
        X_train = X[indices_train]
        self.indices_train = indices_train
        self.indices_test = indices_test
        categorical_features = np.ones(len(features)) == 0

        for i in range(len(features)):
            if ' - ' in features[i]:
                categorical_features[i] = True
        sm = SMOTENC(random_state=42, categorical_features=categorical_features, sampling_strategy = sampling_rate)
        #sm = SMOTE(random_state=random_state)
        output = sm.fit_resample(X_train, y_train)
        X_train = output[0]
        y_train = output[1]
        for i, j in enumerate(indices_train):
            if j in new_label:
                y_train[i] = new_label[j]

        for i, j in enumerate(indices_test):
            if j in new_label:
                y_test[i] = new_label[j]

        self.data_table = data_table
        # X_test = X[indices_test]
        X_test, y_test = self.transform_data(self.test_data_table)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y

        self.check_columns(data_table, self.target)


    def init_data(self, sampling_rate = 1):
        self.init_data_original(sampling_rate)

        all_data = []
        for k in conds:
            csv_path = os.path.join(data_path, f'case1_credit_card/supplement/{k}_current.csv')
            new_data = pd.read_csv(csv_path)
            all_data.append(new_data)

        new_data = pd.concat(all_data, axis = 0)
        X_new, y_new = self.transform_data(new_data)
        self.X_train = np.concatenate((X_new, self.X_train, ), axis = 0)
        self.y_train = np.concatenate((y_new, self.y_train, ), axis = 0)


if __name__ == '__main__':
    from scalable.model_utils import ModelUtil
    util = ModelUtil(data_name = 'credit3', model_name = 'random forest')
    X, y = util.get_rule_matrix()
    y = y.astype(int)
    detect = LRAnomalyDetection(X, y)
    old_score = detect.score()
    old_model = util.model

    model = Model('random forest')
    model.init_data_original()
    model.train()
    model.model_id = -1

    all_data = []
    for k in conds:
        new_data = pd.read_csv(f'data/case1_credit_card/supplement/{k}_current.csv')
        all_data.append(new_data)

    new_data = pd.concat(all_data, axis = 0)
    X_train, y_train = model.transform_data(new_data)
    X_train = np.concatenate((X_train, model.X_train), axis = 0)
    y_train = np.concatenate((y_train, model.y_train), axis = 0)
    model.clf.fit(X_train, y_train)
    print('-' * 20)
    print('new')
    model.get_performance(X_train, y_train, model.X_test, model.y_test)
    model.data_table = pd.concat((model.transform_table(new_data), model.data_table), axis = 0)
    model.model_id = -1
    util.model = model
    model.generate_path()
    X, y = util.get_rule_matrix()
    y = y.astype(int)
    detect = LRAnomalyDetection(X, y)
    new_score = detect.score(X, y)

    print(f'{old_score.mean()} -> {new_score.mean()}')
    mean1 = np.mean(new_score)
    mean2 = np.mean(old_score)
    n1 = len(new_score)
    n2 = len(old_score)

    std1 = np.std(new_score, ddof=1)
    std2 = np.std(old_score, ddof=1)

    t_observed = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))

    alpha = 0.05  # 或其他所需的显著性水平
    # 计算自由度
    df = n1 + n2 - 2
    # 计算p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_observed), df))
    if p_value < alpha:
        print("拒绝原假设，两个样本的均值存在显著差异。")
    else:
        print("无足够证据拒绝原假设，两个样本的均值没有显著差异。")
    print(f'p-value: {p_value}')

