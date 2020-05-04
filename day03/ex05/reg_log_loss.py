#!/usr/bin/env python3

import numpy as np
from sigmoid import sigmoid_

def reg_log_loss_(y_true, y_pred, m, theta, lambda_, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
        m: the length of y_true (should also be the length of y_pred)
        lambda_: a float for the regularization parameter
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    y = y_true
    h = y_pred
    j = (1/m) * np.sum((-y.T)@np.log(h + eps)- (1 - y).T@(1 - np.log(h + eps))
                + (np.dot(lambda_, theta) * theta))
    return j

if __name__ == "__main__":
    x_new = np.arange(1, 13).reshape((3, 4))
    y_true = np.array([1, 0, 1])
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    h = x_new.dot(theta)
    y_pred = sigmoid_(h)
    m = len(y_true)
    print(reg_log_loss_(y_true, y_pred, m, theta, 0.0)) 

    x_new = np.arange(1, 13).reshape((3, 4))
    y_true = np.array([1, 0, 1])
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    y_pred = sigmoid_(np.dot(x_new, theta))
    m = len(y_true)
    print(reg_log_loss_(y_true, y_pred, m, theta, 0.5)) 