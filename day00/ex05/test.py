#! /usr/bin/env python3
import numpy as np
from mat_vec_prod import mat_vec_prod


if __name__ == "__main__":
    import numpy as np
    W = np.array([
        [-8,   8,  -6,  14,  14,  -9,  -4],
        [2, -11,  -2, -11,  14,  -2,  14],
        [-13,  -2,  -5,   3,  -8,  -4,  13],
        [2, 13, -14, -15, -14, -15,  13],
        [2, -1,  12,   3,  -7,  -3,  -6]])
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7, 1))

    print(mat_vec_prod(W, X))
    print(W.dot(X))

    print()
    print(mat_vec_prod(W, Y))
    print(W.dot(Y))
