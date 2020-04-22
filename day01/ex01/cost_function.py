#! /usr/bin/env python3

import numpy as np


def cost_elem_(theta, X, Y):
    if type(X) != np.ndarray or type(Y) != np.ndarray or \
            type(theta) != np.ndarray:
        return None
    if X.shape[1] + 1 != theta.shape[0] or X.shape[0] != Y.shape[0]:
        return None
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((ones, X.reshape(X.shape)), axis=1)
    ret = X[:, 0].reshape(ones.shape)
    for i in range(Y.size):
        ret[i] = ((X[i].dot(theta) - Y[i]) ** 2) * (1/(2 * Y.size))
    return ret


def cost_(theta, X, Y):
    return np.sum(cost_elem_(theta, X, Y))


if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    theta1 = np.array([[2.], [4.]])

    print(cost_elem_(theta1, X1, Y1))
    print(cost_(theta1, X1, Y1))
    print()
    X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    theta2 = np.array([[0.05], [1.], [1.], [1.]])
    Y2 = np.array([[19.], [42.], [67.], [93.]])
    print(cost_elem_(theta2, X2, Y2))
    print(cost_(theta2, X2, Y2))

# Bonus: Cost functions
# https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
