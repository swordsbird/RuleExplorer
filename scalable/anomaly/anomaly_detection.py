
import numpy as np
from sklearn.linear_model import LogisticRegression

class LRAnomalyDetection():
    def __init__(self, X, y, n_classes = 2):
        self.X = X
        self.y = y.astype(np.int32)
        self.label = np.copy(self.y)
        self.n_classes = n_classes
        self.labeled_data = {}
        self.lr = LogisticRegression(class_weight='balanced')
        self.lr.fit(X, y)
        # print('detection', X)
        # print('detection score', self.lr.score(X, y))
        self.w = self.lr.coef_[0]
        self.intercept = self.lr.intercept_

    def compute_loss(self, y_true, y_pred):
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        losses = (y_zero_loss + y_one_loss) * self.current_weight
        return -np.mean(losses)

    def pred(self):
        ret = 1 / (1 + np.exp(-(self.X * self.w).sum(axis = 1) - self.intercept))
        return ret

    def score(self, X = None, y = None):
        if X is None and y is None:
            X = self.X
            y = self.y
        n_rules = len(y)
        #if self.n_classes > 2:
        yp = self.lr.predict_proba(X)
        yh = np.ones((n_rules, self.n_classes)) / 2
        for i, j in enumerate(y):
            yh[i, j] = yp[i].max()
        d = (yp - yh) * 2
        d = np.maximum(d.max(axis = 1), 0)
        return d
        #else:
        #    d = 1 / (1 + np.exp((-(X * self.w).sum(axis = 1) - self.intercept) * 2))
        #    d = 1 - np.array([s if y[i] == 1 else 1 - s for i, s in enumerate(d)])
        #    return d

    def weight(self):
        labels = np.unique(self.y)
        class_count = [(self.y == i).sum() for i in labels]
        max_count = np.max(class_count)
        weight = np.array([max_count / class_count[i] for i in self.y])
        if len(self.labeled_data) > 0:
            for i in self.labeled_data:
                weight[i] = 1
            labeled_weight = len(self.labeled_data)
            unlabeled_weight = np.sum(weight) - labeled_weight
            for i in self.labeled_data:
                weight[i] *= (unlabeled_weight / labeled_weight) * 1
            weight /= unlabeled_weight
        return weight
    
    def adjust_weight(self, idx, anomaly):
        if anomaly:
            self.y[idx] = 1 - self.label[idx]
        else:
            self.y[idx] = self.label[idx]
        self.labeled_data[idx] = anomaly
        self.fit()
    
    def compute_grads(self, x, y_true, y_pred):
        difference =  (y_pred - y_true) * self.current_weight
        grad_b = np.mean(difference)
        grads_w = np.matmul(x.transpose(), difference)
        grads_w = np.array([np.mean(grad) for grad in grads_w])
        return grads_w, grad_b

    def fit(self, epochs = 250, eta = 2e-3):
        self.current_weight = self.weight()
        for i in range(epochs):
            pred = self.pred()
            loss = self.compute_loss(self.y, pred)
            if i % 20 == 0:
                print(f'{i} epochs, loss: {loss}')
            dw, db = self.compute_grads(self.X, self.y, pred)
            self.w -= eta * dw
            self.intercept -= eta * db