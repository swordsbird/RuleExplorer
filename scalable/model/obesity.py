from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
from scalable.model.base_model import BaseModel
from scalable.config import data_path

random_state = 11

cleanup_nums = {
    "Gender": {"Male": 0, "Female": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "MTRANS": {"Walking": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Automobile": 4},
    "NObeyesdad": {
        "Insufficient_Weight":0,
        "Normal_Weight": 1,
        "Overweight_Level_I": 2,
        "Overweight_Level_II": 3,
        "Obesity_Type_I": 4,
        "Obesity_Type_II": 5,
        "Obesity_Type_III": 6,
    }
}

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'obesity'
        self.data_path = os.path.join(data_path, 'obesity.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'NObeyesdad'
        self.output_labels = ["Insufficient_Weight",
			"Normal_Weight",
			"Overweight_Level_I",
			"Overweight_Level_II",
			"Obesity_Type_I",
			"Obesity_Type_II",
			"Obesity_Type_III"
		]
        self.model_name = model_name
        self.model_id = 1
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 9,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 150,
                'learning_rate': 0.03,
                'num_leaves': 50,
                'max_depth': 6,
                'random_state': random_state,
            }
            
    def transform_data(self, data):
        data = data.replace(cleanup_nums)
        return data.drop(self.target, axis = 1), None

    def init_data(self):
        data_table = self.data_table
        data_table = data_table.replace(cleanup_nums)
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
    # model = Model('random forest')
    model.init_data()
    model.train()
    model.get_performance()
    model.generate_path()
