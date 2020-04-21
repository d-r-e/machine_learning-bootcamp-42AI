#!/usr/bin/env python3

import numpy as np

def vec_gradient(x, y, theta):
    if type(x) != np.ndarray or type(y)  != np.ndarray or \
            type(theta) != np.ndarray:
        return(-1)
    if type(x) != type(y) or type(theta) != type(x):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] != theta.shape[0]:
        return None
    return x.transpose().dot(x.dot(theta) - y) * 1 / y.size

if __name__ == "__main__":
    X = np.array([
        [ -6,  -7,  -9],
            [ 13,  -2,  14],
            [ -7,  14,  -1],
            [ -8,  -4,   6],
            [ -5,  -9,   6],
            [  1,  -5,  11],
            [  9, -11,   8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    Z = np.array([3, 0.5, -6])

    print(vec_gradient(X, Y, Z))
    W = np.array([0,0,0])
    print(vec_gradient(X, Y, W))