#!/usr/bin/env python3

import numpy as np

def regularization(theta, lambda_):
    """
    Computes the regularization term of a non-empty numpy.ndarray with a for-loop.
    Args:
        theta: has to be numpy.ndarray, a vector of dimensions n*1.
        lambda_: has to be a float.
    Returns:
        The regularization term of theta.
        None if theta is empty.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if np.size == 0:
            return None
        return lambda_ * np.sum(theta ** 2)
    except Exception:
        return None

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(regularization(X, 0.3))
    print(regularization(X, 0.01))
    print(regularization(X, 0))