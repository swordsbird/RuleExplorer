import optuna  # pip install optuna
from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
sys.path.append('.')
from scalable.model.credit_card import Model

model = Model('random forest')
model.init_data()
X_test = model.X_test
y_test = model.y_test
X_train = model.X_train
y_train = model.y_train

random_state = 42

def objective(trial, X_train, y_train, X_test, y_test):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_int("max_features", 3, 15),
    }
    model = RandomForestClassifier(**param_grid, random_state = random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    cv_scores = accuracy_score(y_test, preds)
    return cv_scores

study = optuna.create_study(direction="maximize", study_name="RandomForestClassifier")
func = lambda trial: objective(trial, X_train, y_train, X_test, y_test)
study.optimize(func, n_trials=150)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = []
precision = []
recall = []
f1 = []
model = RandomForestClassifier(**study.best_params, random_state = random_state)
model.fit(X_train, y_train)
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
