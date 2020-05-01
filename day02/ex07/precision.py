#! /usr/bin/env python3

import numpy as np
from sklearn.metrics import precision_score, accuracy_score

def precision_score_(y_true, y_pred, pos_label=1):
    try:
        tp = 0
        fp = 0
        for i, elem in enumerate(y_true):
            if y_pred[i] == pos_label and y_true[i] == pos_label:
                tp += 1
            if y_pred[i] == pos_label and y_true[i] != pos_label:
                fp += 1
        return (tp / (tp + fp))
    except Exception:
        return None


if __name__ == "__main__":
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(precision_score_(y_true, y_pred))
    print(precision_score(y_true, y_pred))

    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print(precision_score_(y_true, y_pred, pos_label='dog'))
    print(precision_score(y_true, y_pred, pos_label='dog'))

    print(precision_score_(y_true, y_pred, pos_label='norminet'))
    print(precision_score(y_true, y_pred, pos_label='norminet'))