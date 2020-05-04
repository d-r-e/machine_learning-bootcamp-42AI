#!/usr/bin/env python3
import numpy as np


def reg_linear_grad(y, x, theta, lambda_):
    m = y.size
    grad = np.zeros(theta.shape[0] - 1)
    ones = np.ones(x.shape[0]).reshape(-1, 1)
    x = np.concatenate((ones, x), axis=1)
    h = x.dot(theta)
    grad[0] = (1/m) * np.sum((h - y) * x[:, 1])
    for i in range(1, x.shape[1] - 1):
        grad[i] = (1/m) * (np.sum((h - y) * x[:, i + 1]) + lambda_*theta[i])
    return grad


if __name__ == "__main__":
    X = np.array([
        [-6,  -7,  -9],
        [13,  -2,  14],
        [-7,  14,  -1],
        [-8,  -4,   6],
        [-5,  -9,   6],
        [1,  -5,  11],
        [9, -11,   8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta = np.array([0, 3, 10.5, -6])

    grad = reg_linear_grad(Y, X, theta, 1)
    print(grad)
    print(reg_linear_grad(Y, X, theta, 0.5))
    print(reg_linear_grad(Y, X, theta, 0.0))
