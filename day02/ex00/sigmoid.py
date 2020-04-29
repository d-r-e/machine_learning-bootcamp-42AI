#! /usr/bin/env python3
import numpy as np


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = -4
    print(sigmoid_(x))
    x = 2
    print(sigmoid_(x))
    x = np.array([-4, 2, 0])
    print(sigmoid_(x))
