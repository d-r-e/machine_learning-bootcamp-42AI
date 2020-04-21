#! /usr/bin/env python3

import numpy as np

def sum_(x, f):
    if type(x) != np.ndarray or len(x) < 1:
        return None
    try:
        data = map(f, x)
        ret = 0
        for each in list(data):
            ret += each
        return float(ret)
    except TypeError:
        return None

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(sum_(X, lambda x:x))
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(sum_(X, lambda x: x**2))
    print(sum_(X, str.upper))
    print(sum_((), lambda x:x))