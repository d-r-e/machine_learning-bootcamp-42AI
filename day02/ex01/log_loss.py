#! /usr/bin/env python3

import numpy as np
from sigmoid import sigmoid_


def log_loss_(y_true, y_pred, m, eps=1e-15):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if m != y_true.size or m != y_pred.size:
        return None
    sigma = np.zeros(m)
    sigma = y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)
    J = (-1 / m) * (np.sum(sigma))
    return J


if __name__ == "__main__":
    x = 4
    y_true = 1
    theta = 0.5
    y_pred = sigmoid_(x * theta)
    m = 1   # length of y_true is 1
    print(log_loss_(y_true, y_pred, m))

    x = [1, 2, 3, 4]
    y_true = 0
    theta = [-1.5, 2.3, 1.4, 0.7]
    x_dot_theta = sum([a*b for a, b in zip(x, theta)])
    y_pred = sigmoid_(x_dot_theta)
    m = 1
    print(log_loss_(y_true, y_pred, m))

    x_new = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    y_true = [1, 0, 1]
    theta = [-1.5, 2.3, 1.4, 0.7]
    x_dot_theta = []
    for i in range(len(x_new)):
        my_sum = 0
        for j in range(len(x_new[i])):
            my_sum += x_new[i][j] * theta[j]
        x_dot_theta.append(my_sum)
    y_pred = sigmoid_(x_dot_theta)
    m = len(y_true)
    print(log_loss_(y_true, y_pred, m))
