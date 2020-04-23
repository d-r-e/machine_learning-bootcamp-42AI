# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    mylinearregression.py                             :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: darodrig <darodrig@42madrid.com>          +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2020/04/23 18:57:26 by darodrig         #+#    #+#              #
#    Updated: 2020/04/23 18:57:26 by darodrig        ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import numpy as np


class MyLinearRegression():
    """
    Description:
        My personal linear regression class to fit like a boss.
    """
    def __init__(self, theta):
        self.theta = np.array(theta)

    def predict_(self, X):
        theta = self.theta
        if type(X) != np.ndarray:
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

    def cost_elem_(self, X, Y):
        theta = self.theta
        if type(X) != np.ndarray or type(Y) != np.ndarray or \
                type(theta) != np.ndarray:
            return None
        if X.shape[1] + 1 != theta.shape[0] or X.shape[0] != Y.shape[0]:
            return None
        if X.size == 0 or Y.size == 0 or theta.size == 0:
            return None
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X.reshape(X.shape)), axis=1)
        ret = X[:, 0].reshape(ones.shape)
        for i in range(Y.size):
            ret[i] = ((X[i].dot(theta) - Y[i]) ** 2) * (1/(2 * Y.size))
        return ret

    def cost_(self, X, Y):
        return np.sum(self.cost_elem_(X, Y))

    def fit_(self, X, Y, alpha, n_cycle):
        theta = self.theta
        if type(theta) != np.ndarray or theta.size == 0:
            return None
        if type(X) != np.ndarray or X.size == 0:
            return None
        if type(Y) != np.ndarray or Y.size == 0:
            return None
        ones = np.ones((X.shape[0], 1))
        n = theta.size
        s = Y.size
        X = np.concatenate((ones, X), axis=1)
        for i in range(0, n_cycle):
            h0 = np.sum((np.dot(X, theta) - Y))
            theta[0] = theta[0] - ((alpha / X.shape[0]) * h0)
            for j in range(1, n):
                hn = np.sum((np.dot(X, theta) - Y).T * (X[:, j]).reshape(1, s))
                theta[j] = theta[j] - ((alpha / X.shape[0]) * hn)
        self.theta = theta
        return theta
