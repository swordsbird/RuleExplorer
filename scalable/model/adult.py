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
        self.data_name = 'adult'
        self.data_path = os.path.join(data_path, 'adult.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'income'

        self.model_name = model_name
        self.model_id = -1
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': random_state,
            }

    def init_data(self, sampling_rate = 1):
        df = self.data_table
        df.replace('?', 0, inplace=True)

        education_order = {
            'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4,
            '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9,
            'Assoc-voc': 10, 'Assoc-acdm': 11, 'Prof-school': 12, 'Bachelors': 13,
            'Masters': 14, 'Doctorate': 15
        }
        workclass_order = {
            'Never-worked': 0, 'Without-pay': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3,
            'Private': 4, 'Local-gov': 5, 'State-gov': 6, 'Federal-gov': 7
        }
        marital_status_order = {
            'Never-married': 0, 'Married-spouse-absent': 1, 'Separated': 2,
            'Divorced': 3, 'Widowed': 4, 'Married-civ-spouse': 5, 'Married-AF-spouse': 6
        }
        occupation_order = {
            'Priv-house-serv': 0, 'Other-service': 1, 'Handlers-cleaners': 2, 'Farming-fishing': 3,
            'Machine-op-inspct': 4, 'Transport-moving': 5, 'Craft-repair': 6, 'Adm-clerical': 7,
            'Sales': 8, 'Tech-support': 9, 'Protective-serv': 10, 'Prof-specialty': 11,
            'Exec-managerial': 12, 'Armed-Forces': 13
        }
        relationship_order = {
            'Own-child': 0, 'Other-relative': 1, 'Not-in-family': 2,
            'Unmarried': 3, 'Wife': 4, 'Husband': 5
        }

        income_order = {
            '<=50K': 0, '>50K': 1
        }

        df['education'] = df['education'].replace(education_order)
        df['relationship'] = df['relationship'].replace(relationship_order)
        df['occupation'] = df['occupation'].replace(occupation_order)
        df['marital.status'] = df['marital.status'].replace(marital_status_order)
        df['workclass'] = df['workclass'].replace(workclass_order)
        df['income'] = df['income'].replace(income_order)

        country_group = {
            # Group 1: United States and countries with similar high-income economies and global influence.
            'United-States': 1, 'Canada': 1, 'England': 1, 'Germany': 1, 'Ireland': 1, 
            'France': 1, 'Holand-Netherlands': 1, 'Italy': 1, 'Scotland': 1,
            
            # Group 2: Emerging economies with significant growth, industrialization, and middle-income status.
            'Philippines': 2, 'India': 2, 'China': 2, 'Taiwan': 2, 'Japan': 2, 
            'South': 2, 'Iran': 2, 'Hong': 2,
            
            # Group 3: Developing countries, generally considered lower-income but with varying levels of development.
            'Mexico': 3, 'Puerto-Rico': 3, 'El-Salvador': 3, 'Cuba': 3, 'Jamaica': 3,
            'Dominican-Republic': 3, 'Guatemala': 3, 'Columbia': 3, 'Haiti': 3,
            'Nicaragua': 3, 'Peru': 3, 'Ecuador': 3, 'Trinadad&Tobago': 3,
            
            # Group 4: Countries with less industrialization and those not fitting neatly into the other three categories.
            'Vietnam': 4, 'Thailand': 4, 'Cambodia': 4, 'Laos': 4, 'Yugoslavia': 4,
            'Poland': 4, 'Hungary': 4, 'Portugal': 4, 'Greece': 4,
            'Outlying-US(Guam-USVI-etc)': 4, 'Honduras': 4
        }

        df['native.country'] = df['native.country'].replace(country_group)
        race_dummies = pd.get_dummies(df['race'], prefix='race')
        gender_dummies = pd.get_dummies(df['sex'], prefix='sex')

        # Joining the new one-hot encoded columns back to the original DataFrame
        df = pd.concat([df, race_dummies], axis=1)
        df = pd.concat([df, gender_dummies], axis=1)

        # Drop the original columns
        df.drop('race', axis=1, inplace=True)
        df.drop('sex', axis=1, inplace=True)

        features = df.drop(self.target, axis=1).columns
        X = df.drop(self.target, axis=1).values
        y = df[self.target].values
        categorical_features = [i for i, k in enumerate(features) if '_' in k]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        sm = SMOTENC(random_state=190, categorical_features=categorical_features, sampling_strategy = sampling_rate)
        output = sm.fit_resample(X_train, y_train)
        X_train = output[0]
        y_train = output[1]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
        self.y_test = y_test
        self.X = X
        self.y = y
        self.data_table = df

        self.check_columns(df, self.target)

if __name__ == '__main__':
    model = Model('random forest')
    model.init_data()
    model.train()
    model.get_performance()
    model.generate_path()
