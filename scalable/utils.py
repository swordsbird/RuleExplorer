import numpy as np
from scalable.model_utils import ModelUtil
from scalable.model.utils import get_model
from scalable.anomaly import LRAnomalyDetection

def is_int(s):
    return isinstance(s,int) or (np.isscalar(s) and np.issubsctype(np.asarray(s), np.integer))

def get_trained_model(data_name, model_name):
    model = get_model(data_name, model_name)
    model.init_data()
    model.X_train = model.X_train.astype(np.float64)
    model.X_test = model.X_test.astype(np.float64)
    if not is_int(model.y_train[0]):
        labels = model.output_labels
        relabel = {}
        for i, l in enumerate(labels):
            relabel[l] = i
        model.y_train = np.array([relabel[i] for i in model.y_train])
        model.y_test = np.array([relabel[i] for i in model.y_test])
    model.train()
    return model

def generate_model_paths(dataset, model_name):
    modelutil = ModelUtil(data_name = dataset, model_name = model_name)
    model = modelutil.model
    X, y = modelutil.get_rule_matrix()
    y = y.astype(int)
    idx = [i for i, p in enumerate(model.paths) if p['coverage'] > 0.01]
    res = LRAnomalyDetection(X[idx], y[idx], model.n_classes)
    score = res.score(X, y)

    '''
    feature_importance = []
    for i in range(len(model.data_table.columns)):
        j = modelutil.feature_pos[i][1]
        feature_importance.append((model.data_table.columns[j], res.w[i]))
    feature_importance = sorted(feature_importance, key = lambda x: -x[1])
    '''
    for i, val in enumerate(score):
        model.paths[i]['score'] = val
        model.paths[i]['cost'] = val
        model.paths[i]['feature_vector'] = X[i] * np.abs(res.w)
        model.paths[i]['X'] = X[i]
        model.paths[i]['y'] = y[i]
    print('average score', np.mean(score))
    return modelutil
