#! /usr/bin/env python3
import numpy as np


def mse(y, y_hat):
    if type(y) != np.ndarray or type(y_hat) != np.ndarray:
        return None
    if y.shape != y_hat.shape or len(y.shape) != 1:
        return None
    return np.sum((y - y_hat) ** 2) / y.size


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])

    print(mse(X, Y))
    print(mse(X, X))
