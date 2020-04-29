#! /usr/bin/env python3
import numpy as np
from sigmoid import sigmoid_


def log_gradient_(x, y_true, y_pred):
    x = np.array(x)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = []
    if y_true.size > 1:
        x = np.transpose(x)
        for i, row in enumerate(x):
            sigma.append(np.sum((y_pred - y_true).dot(x[i])))
        return sigma
    for i, row in enumerate(x):
        sigma.append(np.sum((y_pred - y_true) * x[i]))
    return np.array(sigma)


if __name__ == "__main__":
    x = [1, 4.2]  # 1 represent the intercept
    y_true = 1
    theta = [0.5, -0.5]
    x_dot_theta = sum([a*b for a, b in zip(x, theta)])
    y_pred = sigmoid_(x_dot_theta)
    print(log_gradient_(x, y_pred, y_true))
    x = [1, -0.5, 2.3, -1.5, 3.2]
    y_true = 0
    theta = [0.5, -0.5, 1.2, -1.2, 2.3]
    x_dot_theta = sum([a*b for a, b in zip(x, theta)])
    y_pred = sigmoid_(x_dot_theta)
    print(log_gradient_(x, y_true, y_pred))
    x_new = [[1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [1, 10, 11, 12, 13]]
# first column of x_new are intercept values initialized to 1
    y_true = [1, 0, 1]
    theta = [0.5, -0.5, 1.2, -1.2, 2.3]
    x_new_dot_theta = []
    for i in range(len(x_new)):
        my_sum = 0
        for j in range(len(x_new[i])):
            my_sum += x_new[i][j] * theta[j]
        x_new_dot_theta.append(my_sum)
    y_pred = sigmoid_(x_new_dot_theta)
    print(log_gradient_(x_new, y_true, y_pred))
