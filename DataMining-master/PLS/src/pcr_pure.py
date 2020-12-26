import numpy as np
from scipy import linalg


def pcr_pure(xtx, xty, x_mean, y_mean, n, p, is_scale=True, is_var_exp=False, yty=None):
    xtx_scale = xtx - np.outer(x_mean, x_mean)*n
    xty_scale = xty - x_mean * y_mean * n
    x_std = None
    if is_scale:
        x_std = np.sqrt(np.diag(xtx_scale)/(n-1))
        x_std_mat = 1/np.repeat(x_std.reshape((1, p)), p, axis=0)
        xtx_scale = x_std_mat.T * xtx_scale * x_std_mat
        xty_scale = xty_scale/x_std
    s, v = linalg.eigh(xtx_scale)
    s = s[-1::-1]
    v = v[:, -1::-1]
    vs = v
    xty_scale = np.dot(v.T, xty_scale) / s
    b1 = (np.cumsum(vs * xty_scale, axis=1)).T
    if is_scale:
        b1 = b1 / x_std
    b0 = (y_mean - np.dot(b1, x_mean)).reshape((p, 1))
    if is_var_exp:
        var_y = yty - n * y_mean * y_mean
        var_exp_x = np.cumsum(s)
        var_exp_x = var_exp_x / var_exp_x[-1]
        var_exp_y = xty_scale * xty_scale * s
        var_exp_y = np.cumsum(var_exp_y) / var_y
        return np.c_[b0, b1], var_exp_x, var_exp_y
    else:
        return np.c_[b0, b1]
