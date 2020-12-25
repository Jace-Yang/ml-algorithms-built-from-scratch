import numpy as np
from scipy import linalg


def pcr_pure(xtx, xty, x_mean, y_mean, n, p, is_scale=True, is_var_exp=True, yty=None):
    # centralization
    # np.outer() returns a p*p matrix
    xtx_scale = xtx - np.outer(x_mean, x_mean) * n
    xty_scale = xty - x_mean * y_mean * n
    x_std = None

    # standardization
    if is_scale:
        # diagonal elements of xtx is the variance of x
        x_std = np.sqrt(np.diag(xtx_scale)/(n-1))  # return a column vector of p dimension
        x_std_inverse = 1/np.repeat(x_std.reshape((1, p)), p, axis=0)  # return 1/std
        xtx_scale = x_std_inverse.T * xtx_scale * x_std_inverse
        xty_scale = xty_scale / x_std  # xty and x_std are both column vectors thus can divide directly

    # compute Eigenvectors and values of covariance matrix xtx
    egvalues, egvectors = linalg.eigh(xtx_scale)
    # rearrange decreasingly
    egvalues = egvalues[-1::-1]
    egvectors = egvectors[:, -1::-1]

    # compute coefficients
    xty_scale = np.dot(egvectors.T, xty_scale)
    b = (np.cumsum(egvectors / egvalues * xty_scale, axis=1)).T
    # each row is a case where different nums of PCs are chosen
    # divide by std if is_scale is True
    if is_scale:
        b = b/x_std
    # compute intercept
    b0 = (y_mean-np.dot(b, x_mean)).reshape((p, 1))

    # return proportion of variance if needed
    if is_var_exp:
        var_x_exp = np.cumsum(egvalues)
        var_x_exp = var_x_exp/var_x_exp[-1]
        var_y_total = yty - n*y_mean*y_mean
        var_y_exp = xty_scale * xty_scale / egvalues  # the variance explained by each PC
        var_y_exp = np.cumsum(var_y_exp) / var_y_total

        # output
        return np.c_[b0, b], var_x_exp, var_y_exp
    else:
        # output
        return np.c_[b0, b]
