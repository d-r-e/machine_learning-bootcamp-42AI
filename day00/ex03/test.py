#! /usr/bin/env python3

import numpy as np
from std import std

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    if np.std(X) == std(X):
        if np.std(X/2) == std(X/2):
            print("OK")
    else:
        print("FAIL")
