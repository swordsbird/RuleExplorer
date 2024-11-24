
import numpy as np
import tempfile
from numpy.lib.arraysetops import unique
import pysbrl

def path_coverage(path, X):
    ans = np.ones(len(X))
    m = path.get('range')
    for key in m:
        ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
    return ans

class SBRLResult():
    def __init__(self, paths, ids, outputs, default):
        self.paths = []
        for i in range(len(ids)):
            self.paths.append({
                'original': paths[ids[i]],
                'range': paths[ids[i]]['range'],
                'output': paths[ids[i]]['value'] > 0,#np.argmax(outputs[i]),
            })
        self.default = default

def make_sbrl(paths, X, y, lam = 20, eta = 2.0):
    data_file = tempfile.NamedTemporaryFile("w", delete=False)
    label_file = tempfile.NamedTemporaryFile("w", delete=False)
    data_file.write("n_items: %s\n"%len(paths))
    data_file.write("n_samples: %s\n"%len(X))
    for i, path in enumerate(paths):
        s = '\{Rule%s\}\t '%(i)
        bits = path_coverage(path, X)
        s = s + ' '.join([str(int(b)) for b in bits]) + '\n'
        data_file.write(s)
    classes = np.unique(y)
    label_file.write("n_items: %s\n"%len(classes))
    label_file.write("n_samples: %s\n"%len(X))
    for value in classes:
        s = '\{label=%s\}\t '%(value)
        bits = (y == value)
        s = s + ' '.join(['1' if b else '0' for b in bits]) + '\n'
        label_file.write(s)
    data_file.close()
    label_file.close()
    rule_ids, outputs, rule_str = pysbrl.train_sbrl(data_file.name, label_file.name, lam, n_chains=50, eta=1, max_iters=20000)
    #for i in range(min(10, len(rule_ids))):
    #    print(outputs[i], paths[rule_ids[i]]['value'], paths[rule_ids[i]]['confidence'])
    all_bits = np.zeros(len(X))
    for i in rule_ids:
        bits = path_coverage(paths[i], X)
        all_bits = all_bits + bits
    unique_y = np.unique(y)
    default_output = 0
    default_count = 0
    all_uncover_bits = (all_bits == 0)
    for output in unique_y:
        curr_count = np.sum(all_uncover_bits * (y == output))
        if curr_count > default_count:
            default_count = curr_count
            default_output = output
    print('all_uncover_bits', np.sum(all_uncover_bits), len(all_uncover_bits))
    return SBRLResult(paths, rule_ids, outputs, default_output)

def predict(X, sbrl):
    left = np.ones(X.shape[0])
    Y = np.zeros(X.shape[0])
    for p in sbrl.paths:
        ans = np.ones(X.shape[0])
        m = p.get('range')
        for key in m:
            ans = ans * (X[:,int(key)] >= m[key][0]) * (X[:,int(key)] < m[key][1])
        current = left * ans
        if p.get('output') > 0:
            Y += current
        left -= current
    Y += left * sbrl.default
    Y = np.where(Y > 0, 1, 0)
    return Y, np.sum(left)

def test_sbrl(sbrl, X, y, log = False):
    y1, not_covered = predict(X, sbrl)
    if log:
        print(not_covered)
    return np.sum((y == y1)) / len(y1)
