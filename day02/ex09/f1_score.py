#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import f1_score
from precision import precision_score_
from recall import recall_score_


def f1_score_(y_true, y_pred, pos_label=1):
    """
    Compute the f1 score.
    F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
        pos_label: str or int, the class on which to report
        the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        prec = precision_score_(y_true, y_pred, pos_label)
        recall = recall_score_(y_true, y_pred, pos_label)
        return 2 * (prec * recall / (prec + recall))
    except Exception:
        return None


if __name__ == "__main__":
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(f1_score_(y_true, y_pred))
    print(f1_score(y_true, y_pred))
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet',
                       'dog', 'dog', 'dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet',
                       'dog', 'norminet', 'dog', 'norminet'])
    print(f1_score_(y_true, y_pred, pos_label='dog'))
    print(f1_score(y_true, y_pred, pos_label='dog'))
    # 0.6666666666666665
    # 0.6666666666666665

    # Test n.3
    print(f1_score_(y_true, y_pred, pos_label='norminet'))
    print(f1_score(y_true, y_pred, pos_label='norminet'))
