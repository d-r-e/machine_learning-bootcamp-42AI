#! /usr/bin/env python3
import numpy as np

class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
        self.thetas = []
        self.loss_list =[]
        self.alpha_list = []
        self.threshold = .5

    def fit(self, x_train, y_train):
        """
        Fit the model according to the given training data.
        Args:
        x_train: a 1d or 2d numpy ndarray for the samples
        y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
        self : object
        None on any error.
        Raises:
        This method should not raise any Exception.
        """
        m, n = x_train.shape[:2]
        self.thetas = np.zeros(n)
        gap = int(self.max_iter / 10)

        for i in range(self.max_iter):
            z = np.dot(x_train, self.thetas)
            h = self.__sigmoid_(z)
            gradient = self.__vec_log_gradient_(x_train, y_train, h)
            # gradient = np.dot(x_train.T, (h - y_train)) / y_train.size
            self.thetas -= self.alpha / m * gradient #the most important part

            if self.verbose and i % gap == 0:
                z = np.dot(x_train, self.thetas)
                h = self.__sigmoid_(z)
                print(f'epoch: {i} loss: {self.__vec_log_loss_(y_train, h, y_train.shape[0])} \t')

        return self

    def predict(self, x_train):
        """
        Predict class labels for samples in x_train.
        Arg:
        x_train: a 1d or 2d numpy ndarray for the samples
        Returns:
        y_pred, the predicted class label per sample.
        None on any error.
        Raises:
        This method should not raise any Exception.
        """
        return self.__sigmoid_(np.dot(x_train, self.thetas)) >= self.threshold

    def score(self, x_train, y_train):
        """
        Returns the mean accuracy on the given test data and labels.
        Arg:
        x_train: a 1d or 2d numpy ndarray for the samples
        y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
        Mean accuracy of self.predict(x_train) with respect to y_true
        None on any error.
        Raises:
        This method should not raise any Exception.
        """
        # averages the result of comparing every predicted value to the real one
        return (self.predict(x_train) == y_train).mean()


    def __sigmoid_(self, x):
        """
        Compute the sigmoid of a scalar or a list.
        Args:
        x: a scalar or list
        Returns:
        The sigmoid value as a scalar or list.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        x = np.asarray(x)
        sigm = 1. / (1. + np.exp(-x))
        return sigm

    def __vec_log_gradient_(self, x, y_true, y_pred):
        """
        Compute the gradient.
        Args:
        x: a list or a matrix (list of lists) for the samples
        y_true: a scalar or a list for the correct labels
        y_pred: a scalar or a list for the predicted labels
        Returns:
        The gradient as a scalar or a list of the width of x.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        X = np.array(x)
        y_t, y_p = np.array(y_true), np.array(y_pred)
        # gradient = np.dot(X.T, (y_p - y_t)) / y_t.shape[0]
        # gradient = ((y_p - y_t) * X.T) / X.shape[0]
        gradient = np.dot((y_p - y_t), X)
        return gradient

    def __vec_log_loss_(self, y_true, y_pred, m, eps=1e-15):
        """
        Compute the logistic loss value.
        Args:
        y_true: a scalar or a list for the correct labels
        y_pred: a scalar or a list for the predicted labels
        m: the length of y_true (should also be the length of y_pred)
        eps: eps (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        y_t, y_p = np.array(y_true), np.array(y_pred)
        cost = (1 / m) * (((-y_t).T * np.log(y_p + eps)) - ((1 - y_t).T * np.log(1 - y_p + eps)))
        # cost = (1 / m) * (((-y_t).T @ np.log(y_p + eps)) - ((1 - y_t).T @ np.log(1 - y_p + eps)))
        # loss = (np.dot(y_train, np.log(np.add(y_pred, eps))) + np.dot((1 - y_train), np.log(1 - y_pred + eps))) * (-1 / m)
        # add fonction loss de jmaisonn (pas forcèment correcte !)
        return cost if isinstance(cost, float) else cost.sum()