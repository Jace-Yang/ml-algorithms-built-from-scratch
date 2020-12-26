import numpy as np


def qf(a_mat, x):
    return np.sum(x * np.dot(a_mat, x))
