#!/usr/bin/env python3
import numpy as np


def vectorized_regularization(theta, lambda_):
    """
    Computes the regularization term of a non-empty numpy.ndarray,
    vector wise.
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
        if theta.size == 0 or type(lambda_) != float:
            return None
        return lambda_ * theta.T.dot(theta)
    except Exception:
        return None


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(vectorized_regularization(X, 0.3))
    print(vectorized_regularization(X, 0.01))
    print(vectorized_regularization(X, 0))
