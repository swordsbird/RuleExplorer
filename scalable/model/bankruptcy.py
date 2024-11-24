from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
from imblearn.over_sampling import SMOTE
from scalable.model.base_model import BaseModel
from scalable.config import data_path

random_state = 10

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'bankruptcy'
        self.data_path = os.path.join(data_path, 'discarded_case_bankruptcy/bank.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'Bankrupt?'
        self.output_labels = ['bankrupt', 'non-bankrupt']

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 500,
                'learning_rate': 0.45,
                'num_leaves': 100,
                'max_depth': 6,
                'lambda_l1': 0.00100121865,
                'lambda_l2': 0.03078951866,
                'bagging_fraction': 0.908,
                'feature_fraction': 0.943,
                'bagging_freq': 4,
                'random_state': random_state,
            }
    
    def init_data(self):
        data_table = self.data_table

        corr = data_table.corr()
        n_cols = len(data_table.columns)
        dropped = []
        for i in range(n_cols):
            k1 = data_table.columns[i]
            for k2 in data_table.columns[i + 1:]:
                if k1 != k2 and abs(corr[k1][k2]) > 0.9:
                    dropped.append(k2)
        dropped = set(dropped)
        dropped = [k for k in dropped]
        # dropped += ['Net Income Flag', 'Liability-Assets Flag']
        data_table = data_table.drop(dropped, axis = 1)

        # data_table['Bankrupt?'] = data_table['Bankrupt?'].apply(lambda x: 1 - x)
        X = data_table.drop(self.target, axis=1).values
        y = data_table[self.target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        sm = SMOTE(random_state=random_state)
        output = sm.fit_resample(X_train, y_train)
        X_train = output[0]
        y_train = output[1]
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y
        self.data_table = data_table

        self.check_columns(data_table, self.target)

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()
    model.train()
    model.get_performance()
    model.generate_path()
