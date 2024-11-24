import numpy as np
from scalable.hierarchy import generate_model_paths

def dumpBoostingTree(root):
    stack = []
    stack.append((root, 0, 0))
    n_nodes = 1
    ret = ''
    nodes = {}
    while len(stack) > 0:
        x, node_id, depth = stack.pop()
        nodes[node_id] = x
        x['depth'] = depth
        if 'decision_type' in x:
            x['left'] = n_nodes
            x['right'] = n_nodes + 1
            stack.append((x['left_child'], n_nodes, depth + 1))
            stack.append((x['right_child'], n_nodes + 1, depth + 1))
            n_nodes += 2
    ret = "NB_NODES: %d\n" % (n_nodes)
    for node_id in range(n_nodes):
        x = nodes[node_id]
        if 'decision_type' not in x:
            ret += "%d LN -1 -1 -1 -1 %d %.4f\n" % (node_id, x['depth'], x['leaf_value'])
        else:
            ret += "%d IN %d %d %d %.4f %d -1\n" % (node_id, x['left'], x['right'], x['split_feature'], x['threshold'], x['depth'])      
    return ret

def dumpBoostingTrees(model, filepath):
    info = model.clf._Booster.dump_model()

    f = open(filepath, 'w')
    f.write(f'DATASET_NAME: {model.data_path}\n')
    f.write(f'ENSEMBLE: RF\n')
    f.write(f'NB_TREES: {len(info["tree_info"])}\n')
    f.write(f'NB_FEATURES: {model.X_train.shape[1]}\n')
    f.write(f'NB_CLASSES: 2\n')
    f.write(f'MAX_TREE_DEPTH: {model.parameters["max_depth"]}\n')
    f.write(f'Format: node / node type(LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n\n')

    for i, tree in enumerate(info['tree_info']):
        s = dumpBoostingTree(tree['tree_structure'])
        f.write(f'[TREE {i}]\n')
        f.write(s)
        f.write('\n')
    f.close()

def dumpDecisionTree(clf, X, y):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    status = []
    status.append(np.ones(shape=len(y), dtype=np.int64))
    labels = np.unique(y)
    
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            while children_left[node_id] >= len(status) or children_right[node_id] >= len(status):
                status.append(None)
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
            status[children_left[node_id]] = status[node_id] * (X[:, feature[node_id]] <= threshold[node_id])
            status[children_right[node_id]] = status[node_id] * (X[:, feature[node_id]] > threshold[node_id])
        else:
            is_leaves[node_id] = True
            idxes = np.flatnonzero(status[node_id])
            max_label = -1
            max_count = 0
            for label in labels:
                cnt = (y[idxes] == label).sum()
                if cnt > max_count:
                    max_count = cnt
                    max_label = label
            status[node_id] = max_label
            

    ret = ''
    ret += "NB_NODES: %d\n" % (n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            ret += "%d LN -1 -1 -1 -1 %d %d\n" % (i, node_depth[i], status[i])
        else:
            ret += "%d IN %d %d %d %.4f %d -1\n" % (i, children_left[i], children_right[i], feature[i], threshold[i], node_depth[i])
    return ret

def dumpRandomForest(model, filepath):
    f = open(filepath, 'w')
    f.write(f'DATASET_NAME: {model.data_path}\n')
    f.write(f'ENSEMBLE: RF\n')
    f.write(f'NB_TREES: {len(model.clf.estimators_)}\n')
    f.write(f'NB_FEATURES: {model.X_train.shape[1]}\n')
    f.write(f'NB_CLASSES: 2\n')
    f.write(f'MAX_TREE_DEPTH: {model.parameters["max_depth"]}\n')
    f.write(f'Format: node / node type(LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n\n')

    for i, tree in enumerate(model.clf.estimators_):
        s = dumpDecisionTree(tree, model.X_train, model.y_train)
        f.write(f'[TREE {i}]\n')
        f.write(s)
        f.write('\n')
    f.close()

if __name__ == '__main__':
    data_name = 'cancer'
    model_name = 'random forest'
    modelutil = generate_model_paths(data_name, model_name)
    model = modelutil.model

    f = open('output.txt', 'w')
    f.write(f'DATASET_NAME: {model.data_path}\n')
    f.write(f'ENSEMBLE: RF\n')
    f.write(f'NB_TREES: {len(model.clf.estimators_)}\n')
    f.write(f'NB_FEATURES: {model.X_train.shape[1]}\n')
    f.write(f'NB_CLASSES: 2\n')
    f.write(f'MAX_TREE_DEPTH: {model.parameters["n_estimators"]}\n')
    f.write(f'Format: node / node type(LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n\n')

    for i, tree in enumerate(model.clf.estimators_):
        s = dumpDecisionTree(tree, model.X_train, model.y_train)
        f.write(f'[TREE {i}]\n')
        f.write(s)
        f.write('\n')
    f.close()
