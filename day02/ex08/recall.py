#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import recall_score


def recall_score_(y_true, y_pred, pos_label=1):
    """
    Compute the recall score. The recall score is the ability to detect all
    the positive examples.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
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
