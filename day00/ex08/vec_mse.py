import numpy as np

def vec_mse(y, y_hat):
    if type(y) != np.ndarray or type(y_hat) != np.ndarray:
        return None
    if y.shape != y_hat.shape or len(y.shape) != 1:
        return None
    return np.dot(y_hat - y, y_hat - y) / y.size

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])

    print(vec_mse(X, Y))
    print(vec_mse(X, X))
    
