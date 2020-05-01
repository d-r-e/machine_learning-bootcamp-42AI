#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import recall_score


def recall_score_(y_true, y_pred, pos_label=1):
    try:
        tp = 0  # true positives
        fn = 0  # false negatives
        for i, elem in enumerate(y_true):
            if y_pred[i] == pos_label and y_true[i] == pos_label:
                tp += 1
            if y_pred[i] != pos_label and y_true[i] == pos_label:
                fn += 1
        return (tp / (tp + fn))
    except Exception:
        return None


if __name__ == "__main__":
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(recall_score_(y_true, y_pred))
    print(recall_score(y_true, y_pred))
