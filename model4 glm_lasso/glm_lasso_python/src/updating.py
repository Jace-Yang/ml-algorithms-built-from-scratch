import numpy as np
from scipy import linalg as sl
import math


def updating(left_mat, xxk, xkxk):
    k, _ = np.shape(left_mat)
    lk = sl.solve_triangular(left_mat, xxk, lower=True)
    lkk = math.sqrt(xkxk - np.sum(lk * lk))
    left_mat_up = np.append(left_mat, np.zeros(shape=(k, 1)), axis=1)
    left_mat_down = np.append(lk, lkk)
    return np.append(left_mat_up, left_mat_down.reshape(1, k+1), axis = 0)
