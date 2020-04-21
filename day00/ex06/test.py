#!/usr/bin/env python3
from mat_mat_prod import mat_mat_prod
import numpy as np

if __name__ == "__main__":
    
    X = np.array(range(9)).reshape((3,3))
    Y = 2 * X
    print(mat_mat_prod(X, np.identity(3)))
    print()
    print(mat_mat_prod(X, Y))

    
    W = np.array([
        [ -8,   8,  -6,  14,  14,  -9,  -4],
        [  2, -11,  -2, -11,  14,  -2,  14],
        [-13,  -2,  -5,   3,  -8,  -4,  13],
        [  2,  13, -14, -15, -14, -15,  13],
        [  2,  -1,  12,   3,  -7,  -3,  -6]])
    Z = np.array([
        [ -6,  -1,  -8,   7,  -8],
            [  7,   4,   0, -10, -10],
            [  7, -13,   2,   2, -11],
            [  3,  14,   7,   7,  -4],
            [ -1,  -3,  -8,  -4, -14],
            [  9, -14,   9,  12,  -7],
            [ -9,  -4, -10,  -3,   6]])
    print()
    print(mat_mat_prod(W, Z))