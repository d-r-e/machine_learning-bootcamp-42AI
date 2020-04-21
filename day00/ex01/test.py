from mean import mean
import numpy as np

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(mean(X))
    print(mean(X ** 2))