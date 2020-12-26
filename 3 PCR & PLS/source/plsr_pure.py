import numpy as np
from numpy import linalg as nl
from source.quadratic_form import qf


def pls_pure(xtx, xty, x_mean, y_mean, n, p, is_scale=True, is_var_exp=True, yty=None):

    # centralize or standardize
    xtx_scale = xtx - np.outer(x_mean, x_mean)*n
    xty_scale = xty - x_mean*y_mean*n
    if is_scale:
        x_std = np.sqrt(np.diag(xtx_scale)) / (n-1)
        x_std_mat = 1 / np.repeat(x_std.reshape((1, p)), p, axis=0)
        xtx_scale = x_std_mat.T * xtx_scale * x_std_mat
        xty_scale = xty_scale / x_std

    ete = xtx_scale.copy()
    etf = xty_scale.copy()

    # initiate variables
    # explained variance of X and y
    if is_var_exp:
        var_y = yty - y_mean*y_mean*n
        var_x_exp = np.zeros(p)
        var_y_exp = np.zeros(p)
    else:
        var_y = None
        var_x_exp = None
        var_y_exp = None
    # coefficients
    b = []
    # initiate w* = I
    w_mat = np.eye(p)

    # enter the loop to extract p components and other variables
    for i in np.arange(p):
        # the first component
        etf_norm = nl.norm(etf)
        w = etf / etf_norm
        t_norm_2 = qf(ete, w)
        r = etf_norm / t_norm_2
        p1 = np.dot(ete, w) / t_norm_2
        b1 = r * np.dot(w_mat, w)
        b.append(np.copy(b1))
        if is_var_exp:
            var_x_exp[i] = t_norm_2 * np.sum(p1*p1)
            var_y_exp[i] = t_norm_2 * r * r
        # update values for next componet
        if i < (p-1):
            w_tmp = np.eye(p) - np.outer(w, p1)
            w_tmp_t = w_tmp.T
            ete = np.dot(w_tmp_t, np.dot(ete, w_tmp))
            etf = np.dot(w_tmp_t, etf)
            w_mat = np.dot(w_mat, w_tmp)
    b = np.array(b)

    # calculate cumulative explained variance
    if is_var_exp:
        var_x_exp = np.cumsum(var_x_exp)
        var_x_exp = var_x_exp / var_x_exp[-1]
        var_y_exp = np.cumsum(var_y_exp) / var_y

    # calculate cumulative coefficients
    b = np.cumsum(b, axis=0)
    if is_scale:
        b = b / x_std
    b0 = y_mean - np.dot(b, x_mean)
    b = np.c_[b0.reshape((p, 1)), b]
    
    # output
    if is_var_exp:
        return b, var_x_exp, var_y_exp
    else:
        return b


