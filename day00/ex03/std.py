import numpy as np

def std(x):
    if type(x) != np.ndarray or len(x) < 1:
        return 0
    try:
        m = 0
        for each in x:
            m += each
        m = m / len(x)
        ret = 0
        for each in x:
            ret += (each - m) ** 2
        ret = float(ret / len(x)) ** (1/2)
        return ret
    except TypeError:
        return None