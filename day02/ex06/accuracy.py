#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_score_(y_true, y_pred):
    """
    Compute the accuracy score.
    Args:
        y_true: a scalar or a numpy array for the correct labels.
        y_pred: a scalar or a numpy array for the predicted labels.
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if y_true.size != y_pred.size:
            return None
        return (y_true == y_pred).mean()
    except Exception:
        return None


if __name__ == "__main__":
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(accuracy_score_(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print(accuracy_score_(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))