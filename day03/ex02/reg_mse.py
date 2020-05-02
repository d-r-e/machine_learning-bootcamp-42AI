#!/usr/bin/env python3
import numpy as np


def reg_mse(y, x, theta, lambda_):
    if x.size == 0 or y.size == 0:
        return None
    m = y.size
    ones = np.ones(x.shape[0]).reshape(-1, 1)
    x = np.concatenate((ones, x), axis=1)
    h = x.dot(theta)
    mse = (1/m) * ((h - y).T.dot(h - y) + lambda_*theta.T.dot(theta))
    return mse


if __name__ == "__main__":
    X = np.array([
        [-6,  -7,  -9],
        [13,  -2,  14],
        [-7,  14,  -1],
        [-8,  4,   6],
        [-5, -9,   6],
        [1,  -5,  11],
        [9, -11,   8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta = np.array([0, 3, 0.5, -6])

    print(reg_mse(Y, X, theta, 0))
    print(reg_mse(Y, X, theta, 0.1))
    print(reg_mse(Y, X, theta, 0.5))
