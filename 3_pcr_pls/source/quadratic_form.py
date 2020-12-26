import numpy as np


def qf(x_mat, y):
    return np.sum(y * np.dot(x_mat, y))
