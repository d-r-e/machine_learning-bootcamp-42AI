#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y_true, y_pred, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
        labels: optional, a list of labels to index the matrix. This may be used to reorder or select a subset of labels. (default=None)
    Returns: 
        The confusion matrix as a numpy ndarray.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    if labels == None:
        values = list(set(y_true))
    else:
        values = labels
    for i, elem in enumerate(y_true):
        if y_pred[i] == values[1] and y_true[i] == y_pred[i]:
            tp += 1
        elif y_pred[i] == values[1] and y_true[i] != y_pred[i]:
            fp += 1
        elif y_pred[i] == values[0] and y_true[i] == y_pred[i]:
            tn += 1
        elif y_pred[i] == values[0] and y_true[i] != y_pred[i]:
            fn += 1
    matrix = np.array([[tp, fp], [fn, tn]])
    return matrix


if __name__ == "__main__":
    
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])

    print(confusion_matrix_(y_true, y_pred))
    print(confusion_matrix_(y_true, y_pred, labels=['dog','norminet']))