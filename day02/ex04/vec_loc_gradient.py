#! /usr/bin/env python3

import numpy as np
from sigmoid import sigmoid_


def vec_log_gradient_(x, y_true, y_pred):
    """
    Compute the gradient.
    Args:
        x: a 1d or 2d numpy ndarray for the samples
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
    Returns:
        The gradient as a scalar or a numpy ndarray of the width of x.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    x = np.array(x)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(x.shape) > 1:
        return (y_pred - y_true).dot(x)
    return (y_pred - y_true) * x


if __name__ == "__main__":
    x = np.array([1, 4.2])  # x[0] represent the intercept
    y_true = 1
    theta = np.array([0.5, -0.5])
    y_pred = sigmoid_(np.dot(x, theta))
    print(vec_log_gradient_(x, y_pred, y_true))
    # [0.83201839 3.49447722]

    x = np.array([1, -0.5, 2.3, -1.5, 3.2])  # x[0] represent the intercept
    y_true = 0
    theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
    y_pred = sigmoid_(np.dot(x, theta))
    print(vec_log_gradient_(x, y_true, y_pred))
    # [ 0.99999686 -0.49999843  2.29999277 -1.49999528  3.19998994]

    x_new = np.arange(2, 14).reshape((3, 4))
    x_new = np.insert(x_new, 0, 1, axis=1)
    # first column of x_new are now intercept values initialized to 1
    y_true = np.array([1, 0, 1])
    theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
    y_pred = sigmoid_(np.dot(x_new, theta))
    print(vec_log_gradient_(x_new, y_true, y_pred))
    # [0.99994451 5.99988885 6.99983336 7.99977787 8.99972238]
