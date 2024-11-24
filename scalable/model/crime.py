from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE
from scalable.config import data_path
from scalable.model.base_model import BaseModel
from scalable.model.data_encoding import german_credit_encoding
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

random_state = 190

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'crime'
        self.data_path = os.path.join(data_path, 'crime.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'ViolentCrimesPerPop'
        self.model_id = -1

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 10,
                'num_leaves': 100,
                'learning_rate': 0.05,
                'random_state': random_state,
            }

    def init_data(self, sampling_rate = 1):
        df = self.data_table
        df['ViolentCrimesPerPop'] = (df['ViolentCrimesPerPop'] > df['ViolentCrimesPerPop'].median()).astype(int)
        for k in ['communityname', 'county', 'community', 'fold', 'state']:
            df = df.drop(k, axis = 1)
        for k in df.columns:
            if (df[k] == '?').sum() > len(df) * 0.25:
                df = df.drop(k, axis = 1)
        df.replace('?', 0, inplace=True)
        for k in df.columns:
            if df[k].dtype == 'object':
                df[k] = df[k].astype(float)
        X = df.drop(self.target, axis=1).values
        y = df[self.target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y
        self.data_table = df

        self.check_columns(df, self.target)

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()
    model.train()
    model.get_performance()
    model.generate_path()
    feature_importance = model.clf.feature_importances_
    top_10_features_index = feature_importance.argsort()[-10:][::-1]
    feature = model.data_table.drop(model.target, axis=1).columns
    top_10_features_names = [feature[i] for i in top_10_features_index]
    print("Top 10 features:")
    for feature_name in top_10_features_names:
        print(feature_name)
