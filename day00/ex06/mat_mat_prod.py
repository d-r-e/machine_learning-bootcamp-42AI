from dot import dot
import numpy as np


def mat_mat_prod(x, y):
    if type(x) != np.ndarray or type(y) != np.ndarray:
        return None
    if x.shape[1] != y.shape[0]:
        return None
    ret = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            ret[i][j] = dot(x[i], y[:,j])
    return ret
