from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
from scalable.model.base_model import BaseModel
from scalable.config import data_path

random_state = 11

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'cancer'
        self.data_path = os.path.join(data_path, 'cancer.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'diagnosis'
        self.output_labels = ['negative', 'positive']

        self.model_id = -1
        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 8,
                'random_state': 10,
            }
        else:
            self.parameters = {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'num_leaves': 50,
                'max_depth': 5,
                'random_state': 10,
            }


    def init_data(self):
        data_table = self.data_table.drop(['id'], axis=1)

        data_table['diagnosis'] = data_table['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
        X = data_table.drop(self.target, axis=1).values
        y = data_table[self.target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)

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
