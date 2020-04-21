#! /usr/bin/env python3
import numpy as np
from variance import variance

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    if np.var(X) == variance(X):
        if np.var(X/2) == variance(X/2):
            print("OK")
    else:
        print("FAIL")
