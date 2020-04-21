import numpy as np
from dot import dot


def mat_vec_prod(x, y):
    if x.shape[1] != y.shape[0]:
        print(y.shape)
    if type(x) != np.ndarray or type(y) != np.ndarray or len(x.shape) != 2 or \
            x.shape[1] != y.shape[0] or y.shape[1] != 1:
        return None
    res = []
    for row in x:
        res.append(dot(row, y))
    return np.array(res).reshape(len(res), 1)
