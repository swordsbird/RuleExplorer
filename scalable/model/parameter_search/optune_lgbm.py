from tabnanny import verbose
import optuna  # pip install optuna
from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import os

random_state = 10

def objective(trial, X, y):
    # 后面填充
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 10, 300),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),
    }
    cv = KFold(n_splits=4, shuffle=True, random_state=random_state)
    sm = SMOTE(random_state=random_state)

    cv_scores = np.empty(4)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # LGBM建模
        model = LGBMClassifier(objective="binary", **param_grid)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        model.fit(
            X_train,
            y_train,
            verbose= False,
            eval_set=[(X_test, y_test)]
        )
        # 模型预测
        preds = model.predict(X_test)
        # 优化指标logloss最小
        cv_scores[idx] = accuracy_score(y_test, preds)
    return np.mean(cv_scores)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

project_path = '/home/lizhen/projects/extree/exp'
data_table = pd.read_csv(os.path.join(project_path, 'data/cancer.csv'))

labels = data_table['diagnosis']
labels = labels.apply(lambda x: 1 if x == 'M' else 0)
data_table = data_table.drop(['diagnosis', 'id'], axis=1)
X = data_table.values
y = labels.values

study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=200)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cv = KFold(n_splits=4, shuffle=True, random_state=random_state)
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
    model = LGBMClassifier(objective="binary", **study.best_params)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)]
    )
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
