import numpy as np


def sigmoid(x):
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class BCELoss:
    """
    L = -1/N * sum(ygt * log(sigmoid(yi)) + (1-ygt)*log(1-sigmoid(yi)))
    """

    def grad(self, y_pred, y_true):
        return sigmoid(y_pred) - y_true

    def hess(self, y_pred, y_true):
        return dsigmoid(y_pred)

    def raw_pred_to_proba(self, y_pred_raw):
        return sigmoid(y_pred_raw)
