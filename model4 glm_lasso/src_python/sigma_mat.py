import numpy as np


def sigma_ma(p, rho):
    if p == 1:
        return np.array([[1.0]])
    else:
        mat_left = np.power(rho, np.arange(p - 1, 0, -1)).reshape(p - 1, 1)
        mat_below = np.power(rho, np.arange(p - 1, -1, -1)).reshape(1, p)
        mat_above = np.c_[sigma_ma(p - 1, rho), mat_left]
        return np.r_[mat_above, mat_below]

# def sigma_ma(p, rho):
#     sigma_mat = np.zeros(shape=(p, p), dtype=float)
#     sigma_vec = np.power(rho, np.arange(p))
#     sigma_mat[0] = sigma_vec
#     sigma_mat[p-1] = sigma_vec
#     if p > 2:
#         k = p // 2
#         for i in np.arange(1, k, 1):
#             sigma_mat[p][1:] =