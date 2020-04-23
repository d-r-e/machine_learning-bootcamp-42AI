#!/usr/bin/env python3

# ************************************************************************** #
#                                                                            #
#                                                       :::      ::::::::    #
#    fit.py                                           :+:      :+:    :+:    #
#                                                   +:+ +:+         +:+      #
#    By: darodrig <darodrig@42madrid.com>         +#+  +:+       +#+         #
#                                               +#+#+#+#+#+   +#+            #
#    Created: 2020/04/22 16:12:59 by darodrig        #+#    #+#              #
#    Updated: 2020/04/22 16:12:59 by darodrig       ###   ########.fr        #
#                                                                            #
# ************************************************************************** #

import numpy as np


def predict_(theta, X):
    if type(theta) != np.ndarray or type(X) != np.ndarray:
        return None
    if X.shape[1] + 1 != theta.shape[0]:
        print("Incompatible dimension match between X and theta.")
        return None
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((ones, X), axis=1)
    ret = X[:, 0].reshape(ones.shape)
    for i, each in enumerate(ret):
        ret[i] = np.sum(np.dot(X[i], theta))
    return ret


def fit_(theta, X, Y, alpha, n_cycle):
    if type(theta) != np.ndarray or theta.size == 0:
        return None
    if type(X) != np.ndarray or X.size == 0:
        return None
    if type(Y) != np.ndarray or Y.size == 0:
        return None
    ones = np.ones((X.shape[0], 1))
    n = theta.size
    ysz = Y.size
    X = np.concatenate((ones, X), axis=1)
    for i in range(0, n_cycle):
        h0 = np.sum((np.dot(X, theta) - Y))
        theta[0] = theta[0] - ((alpha / X.shape[0]) * h0)
        for j in range(1, n):
            hn = np.sum((np.dot(X, theta) - Y).T * (X[:, j]).reshape(1, ysz))
            theta[j] = theta[j] - ((alpha / X.shape[0]) * hn)
    return theta


if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
    theta1 = np.array([[1.], [1.]])
    theta1 = fit_(theta1, X1, Y1, alpha=0.01, n_cycle=2000)

    print(theta1)
    print(predict_(theta1, X1))
    print()

    X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                  [0.6, 6., 60.], [0.8, 8., 80.]])
    Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta2 = np.array([[42.], [1.], [1.], [1.]])
    theta2 = fit_(theta2, X2, Y2, alpha=0.0005, n_cycle=42000)
    print(theta2)
    print(predict_(theta2, X2))

# theory:
# https://link.medium.com/XcHsmFr7U5
