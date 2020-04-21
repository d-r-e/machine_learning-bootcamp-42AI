#! /usr/bin/env python3

import numpy as np


def dot(x, y):
    if type(x) != np.ndarray or type(x) != type(y) or \
            len(x) < 1 or len(y) != len(x):
        return None
    ret = 0
    for i in range(len(x)):
        ret += x[i] * y[i]
    return ret


if __name__ == "__main__":
    x = np.array(range(3))
    y = np.array(range(3))
    if np.dot(x, y) == dot(x, y):
        print("OK")
    else:
        print("NOT OK")

    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    if np.dot(X, Y) == dot(X, Y):
        if np.dot(X, X) == dot(X, X):
            print("OK")
        else:
            print("NOT OK")
    else:
        print("NOT OK")
