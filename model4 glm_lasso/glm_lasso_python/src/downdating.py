import numpy as np


def _gives_tran(mx, lmx):
    mc = mx[0] / lmx
    ms = mx[1] / lmx
    tran_mat = np.array([[mc, -ms], [ms, mc]])
    return tran_mat


def downdating(left_mat, k):
    p, _ = np.shape(left_mat)
    p = p - 1
    if k > p:
        return "Wrong input of k!"
    left_mat_k = np.delete(left_mat, k, axis=0)
    mk = k
    while mk < p:
        mx = np.copy(left_mat_k[mk, mk:(mk+2)])
        lmx = np.linalg.norm(mx)
        left_mat_k[mk, mk] = lmx
        left_mat_k[mk, mk + 1] = 0
        if mk < p - 1:
            tmp_mat = np.copy(left_mat_k[(mk + 1):p, mk:(mk + 2)])
            tmp_mat = np.dot(tmp_mat, _gives_tran(mx, lmx))
            left_mat_k[(mk + 1):p, mk:(mk + 2)] = tmp_mat
        mk = mk + 1
    return np.delete(left_mat_k, p, axis=1)

# Examape
# x = np.random.normal(size=(200, 5))
# xx = np.dot(x.T, x)
# # np.identity(2)*5
# left_mat = sl.cholesky(xx,  lower=True)
# for k in np.arange(5):
#     print(downdating(left_mat, k))
#     ind = np.arange(5)
#     ind_k = np.delete(ind, k, axis=0)
#     xk = x[:, ind_k]
#     xxk = np.dot(xk.T, xk)
#     print(sl.cholesky(xxk,  lower=True))
