#! /usr/bin/env python3
# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    pred.py                                           :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: darodrig <darodrig@student.42madrid.com>  +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2020/04/22 11:25:40 by darodrig         #+#    #+#              #
#    Updated: 2020/04/22 11:25:40 by darodrig        ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

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


if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])

    print(predict_(theta1, X1))

    X2 = np.array([[1], [2], [3], [5], [8]])
    theta2 = np.array([[2.]])
    print(predict_(theta2, X2))

    X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                  [0.6, 6., 60.], [0.8, 8., 80.]])
    theta3 = np.array([[0.05], [1.], [1.], [1.]])
    print(predict_(theta3, X3))
