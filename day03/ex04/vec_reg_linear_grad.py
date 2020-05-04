#!/usr/bin/env python3

import numpy as np


def vec_reg_linear_grad(y, x, theta, lambda_):
    m =x.shape[0]
    grad = np.zeros(x.shape[1])
    h = x.dot(theta)
    grad[0] = (1/m) * np.sum((h - y) * x[:, 0])
    for j in range(1, x.shape[1]):
        grad[j] = (1/m) * (np.sum((h - y) * x[:, j]) + (lambda_*theta[j]))
    return grad

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
    theta = np.array([3, 10.5 ,-6])

    print(vec_reg_linear_grad(Y, X, theta, 1))
    # array([-192.64285714,  887.5, -679.57142857])

    print(vec_reg_linear_grad(Y, X, theta, 0.5))
    # array([-192.85714286,  886.75, -679.14285714])

    print(vec_reg_linear_grad(Y, X, theta, 0.0))