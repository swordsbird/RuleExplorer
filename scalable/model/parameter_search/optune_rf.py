import optuna  # pip install optuna
from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE

random_state = 190

def objective(trial, X, y):
    # 后面填充
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    sm = SMOTE(random_state=random_state)

    cv_scores = np.empty(4)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(**param_grid)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        X_test, y_test = sm.fit_resample(X_test, y_test)
        preds = model.predict(X_test)
        cv_scores[idx] = accuracy_score(y_test, preds)
    return np.mean(cv_scores)

project_path = '/home/lizhen/projects/extree/exp'
data_table = pd.read_csv(os.path.join(project_path, 'data/cancer.csv'))

labels = data_table['diagnosis']
labels = labels.apply(lambda x: 1 if x == 'M' else 0)
data_table = data_table.drop(['diagnosis', 'id'], axis=1)
X = data_table.values
y = labels.values

study = optuna.create_study(direction="maximize", study_name="RandomForestClassifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=50)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
sm = SMOTE(random_state=random_state)

cv_scores = np.empty(4)
accuracy = []
precision = []
recall = []
f1 = []
for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # LGBM建模
    model = RandomForestClassifier(**study.best_params)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    model.fit(X_train, y_train)
    X_test, y_test = sm.fit_resample(X_test, y_test)
    # 模型预测
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

print('Accuracy Score is', np.mean(accuracy))
print('Precision is', np.mean(precision))
print('Recall is', np.mean(recall))
print('F1 Score is', np.mean(f1))
print(study.best_params)
