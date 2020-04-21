import numpy as np

def mean(x):
    if type(x) != np.ndarray or len(x) < 1:
        return None
    try:
        ret = 0
        for each in x:
            ret += each
        ret = float(ret / len(x))
        return ret
    except TypeError:
        return None