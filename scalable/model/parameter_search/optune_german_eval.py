import optuna  # pip install optuna
from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTENC
import sys
sys.path.append('.')
from scalable.model.german_new import Model

model = Model('random forest')
model.init_data()

random_state = 190
categorical_features = np.ones(51) > 0
categorical_features[[0, 1, 2, 3, 4, 5, 7, 8]] = False

X = model.X
y = model.y
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


cv_scores = np.empty(4)
accuracy = []
precision = []
recall = []
f1 = []

best_param =  {
    'n_estimators': 200,
    'max_depth': 12,
    'random_state': random_state,
    'max_features': 'auto',
    'oob_score': True,
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'smoth_n_neighbors': 5,
}
smoth_n_neighbors = best_param['smoth_n_neighbors']
del best_param['smoth_n_neighbors']
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
sm = SMOTENC(random_state=random_state, categorical_features=categorical_features, k_neighbors=smoth_n_neighbors)

for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(**best_param)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

    break

print('Accuracy Score is', np.mean(accuracy))
print('Precision is', np.mean(precision))
print('Recall is', np.mean(recall))
print('F1 Score is', np.mean(f1))
