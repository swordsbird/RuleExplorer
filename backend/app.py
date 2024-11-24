from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from flask_compress import Compress

import math
import random
import json
import numpy as np
import sys
sys.path.append('..')
from scalable.algorithm.model_reduction import Extractor, path_predict

from dataset import DatasetLoader
session_storage = {}

def get_dataloader(session_id, dataname):
    if session_id not in session_storage:
        session_storage[session_id] = DatasetLoader()
    return session_storage[session_id].get(dataname)

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
compress = Compress()
compress.init_app(app)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return 0 if obj else 1
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@app.route('/api/distribution', methods=["POST"])
def get_distribution():
    data = json.loads(request.get_data(as_text=True))
    id = data['id']
    feature = data['feature']
    dataname = data['dataname']
    session_id = data['session_id']
    loader = get_dataloader(session_id, dataname)
    path = loader.path_dict[id]
    idxes = [i for i in path['sample_id'] if i < len(loader.data_table)]
    if len(idxes) > 200:
        idxes = random.sample(idxes, 200)
    values = loader.data_table[feature][idxes].tolist()
    return json.dumps(values, cls=NpEncoder)

@app.route('/api/data_table', methods=["POST"])
def get_data():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    precision = int(data['precision'])
    session_id = data['session_id']
    loader = get_dataloader(session_id, dataname)
    features = [feature for feature in loader.data_table.columns]
    first_columns = ['id', 'Predict', 'Label']
    features = [feature for feature in features if feature in first_columns] + [feature for feature in features if feature not in first_columns]
    n = 5000
    values = []
    for feature in features:
        if loader.data_table[feature].dtype != np.float64:
            values.append(loader.data_table[feature].values[:n])
        else:
            values.append([round(x, precision) for x in loader.data_table[feature].values[:n]])
    #shap = [loader.shap_values[i].values[:n] for i in range(len(loader.shap_values))]
    response = {
        'features' : features,
        'values': values,
    }
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/samples', methods=["POST"])
def get_samples():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    session_id = data['session_id']
    loader = get_dataloader(session_id, dataname)
    ids = data['ids']
    response = []
    for i in ids:
        response.append({
            'x': loader.model.X[i].tolist(),
            'y': str(loader.model.y[i]),
        })
    return jsonify(response)

@app.route('/api/features', methods=["POST"])
def get_features():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    session_id = data['session_id']
    loader = get_dataloader(session_id, dataname)
    return json.dumps(loader.features, cls=NpEncoder)


@app.route('/api/suggestions', methods=["POST"])
def get_suggestions():
    data = json.loads(request.get_data(as_text=True))
    ids = data['ids']
    dataname = data['dataname']
    session_id = data['session_id']
    target = data['target']
    loader = get_dataloader(session_id, dataname)
    idxes = [loader.path_index[name] for name in ids]
    samples = loader.get_relevant_samples(idxes)
    suggestions = loader.get_feature_hint(idxes, samples, target, 5)
    return json.dumps(suggestions, cls=NpEncoder)


@app.route('/api/explore_rules', methods=["POST"])
def get_explore_rules():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    session_id = data['session_id']
    max_samples = 1000
    loader = get_dataloader(session_id, dataname)
    fathers = data['idxs']

    idxes = []
    father_idxes = []
    dist = {}

    if len(fathers) < 20:
        n = 80 - len(fathers)
    else:
        n = 80

    n_esitimated_paths = 0
    n_neighbors = {}
    for name in fathers:
        j = loader.path_index[name]
        neighbors = loader.paths[j]['children']
        n_esitimated_paths += len(neighbors) + 1
        n_neighbors[name] = len(neighbors)

    if n_esitimated_paths > 10 * n:
        max_neighbors = np.median([n_neighbors[name] for name in fathers])
        max_neighbors = int(max_neighbors)
        for name in fathers:
            n_neighbors[name] = min(max_neighbors, n_neighbors[name])

    all_neighbors = []
    for name in fathers:
        j = loader.path_index[name]
        all_neighbors += loader.paths[j]['children']
        neighbors = loader.paths[j]['children'][:n_neighbors[name]]
        for k in range(len(neighbors)):
            dist[neighbors[k]] = loader.paths[j]['children_dist'][k]
        idxes += [j] + neighbors
        father_idxes.append(j)

    max_child = math.ceil(n / len(father_idxes) * 3)
    idxset = set()
    new_idxes = []
    for i in idxes:
        if i not in idxset:
            new_idxes.append(i)
            idxset.add(i)
    idxes = new_idxes
    sample_subset = loader.get_relevant_samples(father_idxes)
    if len(sample_subset) > max_samples:
        sample_subset = random.sample(sample_subset, max_samples)

    if len(idxes) > n:
        if len(fathers) == 1:
            idxes = idxes[:n]
        else:
            paths = [loader.paths[i] for i in idxes]
            X = loader.model.X[sample_subset]
            y = loader.model.clf.predict(X)
            info = loader.model_info()['info']
            xi = info.get('xi', 0.1)
            lambda_ = info.get('lambda', 0)
            alpha = loader.model.parameters['n_estimators'] * n / len(loader.paths)
            ex = Extractor(paths, X, y, greedy = True)
            groups = []
            last_i = 0
            for i, j in enumerate(idxes):
                if i > 0 and j in father_idxes:
                    groups.append([k for k in range(last_i, i)])
                    last_i = i
            groups.append([k for k in range(last_i, len(idxes))])
            ex.set_group_limit(groups, max_child)
            w, _, _, _ = ex.extract(n, xi * alpha, lambda_)
            fidelity = ex.evaluate(w, X, y)
            print("path candidates", len(paths))
            print("fidelity", fidelity, len(sample_subset), "samples")
            #w[:len(idxes1)] = 1
            for i, j in enumerate(idxes):
                if j in father_idxes:
                    w[i] = 1
            [idx] = np.nonzero(w)
            idxes = [idxes[i] for i in idx]

    loader.set_current_rules(idxes)
    loader.adjust_hierarchy(idxes, all_neighbors)
    idxes = idxes[:n]
    paths = []
    for i in idxes:
        p = loader.get_encoded_path(i)
        if i in father_idxes:
            p['represent'] = True
        paths.append(p)
    samples = loader.get_relevant_samples(idxes)
    positives, total, prob = loader.get_general_info(samples)
    response = {
        'paths': paths,
        'samples': samples,
        'info': {
            'prob': prob,
            'positives': positives,
            'total': total,
        },
    }
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/selected_rules', methods=["POST"])
def get_selected_rules():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    session_id = data['session_id']
    loader = get_dataloader(session_id, dataname)
    response = []
    idxes = [loader.path_index[i] for i in loader.selected_indexes]
    loader.set_current_rules(idxes)
    loader.init_hierarchy()
    paths = []
    for i in idxes:
        paths.append(loader.get_encoded_path(i))
    samples = loader.get_relevant_samples(idxes)
    positives, total, prob = loader.get_general_info()
    response = {
        'paths': paths,
        'samples': samples,
        'info': {
            'prob': prob,
            'positives': positives,
            'total': total,
        },
    }
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/rule_samples', methods=["POST"])
def get_relevant_samples():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    session_id = data['session_id']
    loader = get_dataloader(session_id, dataname)
    names = data['names']
    N = data['N']
    ids = set()
    for name in names:
        for i in loader.paths[loader.path_index[name]]['sample_id']:
            if i not in ids:
                ids.add(i)
    ids = [i for i in ids]
    ids = random.sample(ids, N)
    response = []
    for i in ids:
        response.append({
            'id': i,
            'x': loader.model.X[i].tolist(),
            'y': str(loader.model.y[i]),
            'shap_values': loader.shap_values[i].values,
        })
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/model_info', methods=["POST"])
def get_model_info():
    data = json.loads(request.get_data(as_text=True))
    user_info = request.headers.get('User-Agent') + request.remote_addr
    session_id = hex(abs(hash(user_info)))
    dataname = data['dataname']
    loader = get_dataloader(session_id, dataname)
    resp = {
        'model_info': loader.model_info(),
        'session_id': session_id,
    }
    return json.dumps(resp, cls=NpEncoder)

@app.route('/api/clear_session', methods=["POST"])
def clear_session():
    data = json.loads(request.get_data(as_text=True))
    session_id = data['session_id']
    del session_storage[session_id]
    return { 'success': True }


