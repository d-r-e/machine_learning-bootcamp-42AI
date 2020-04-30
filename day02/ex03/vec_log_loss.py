#! /usr/bin/env python3
import numpy as np
from sigmoid import sigmoid_


def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
        m: the length of y_true (should also be the length of y_pred)
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if m != y_true.size or m != y_pred.size:
            return None
        j = -(1/m) * ((y_true * np.log(y_pred) +
                      (1-y_true) * np.log(1-y_pred)))
        return np.sum(j)
    except Exception:
        return None


if __name__ == "__main__":
    x = 4
    y_true = 1
    theta = 0.5
    y_pred = sigmoid_(x * theta)
    m = 1   # length of y_true is 1
    print(vec_log_loss_(y_true, y_pred, m))

    x = np.array([1, 2, 3, 4])
    y_true = 0
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    y_pred = sigmoid_(np.dot(x, theta))
    m = 1
    print(vec_log_loss_(y_true, y_pred, m))

    x_new = np.arange(1, 13).reshape((3, 4))
    y_true = np.array([1, 0, 1])
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    y_pred = sigmoid_(np.dot(x_new, theta))
    m = len(y_true)
    print(vec_log_loss_(y_true, y_pred, m))
