import numpy as np
from scipy import linalg
from math import log
from src.lars_iter import lars_iter


def _logit_lasso_init(x, xt, y, n, p):
    y_mean = np.mean(y)
    mu = y_mean * np.ones(n, dtype=float)
    w = mu * (1 - mu)
    b = np.zeros(p, dtype=float)
    b[0] = log(y_mean / (1 - y_mean))
    cc = np.dot(xt, y - mu)
    s = np.sign(cc)
    s[0] = 0
    cc_abs = cc * s
    cc_abs[0] = 0
    j = np.argmax(cc * s)
    lamb = cc_abs[j]
    is_active = np.array([True] + [False] * (p - 1))
    is_active_ = np.copy(is_active)
    is_active_[j] = True
    xt_a = xt[is_active_]
    x_a = x[:, is_active_]
    xtx_a = np.dot(xt_a * w, x_a)
    left_mat = linalg.cholesky(xtx_a, lower=True)
    sa = s[is_active_]
    d = linalg.cho_solve((left_mat, True), sa)
    u = np.dot(d, xt[is_active_])
    a = np.dot(xt[~is_active_], u * w)
    gam = np.ones(p, dtype=float) * lamb
    gam[~is_active_] = np.where(a * lamb <= cc[~is_active_],
                                (lamb - cc[~is_active_]) / (1 - a),
                                (lamb + cc[~is_active_]) / (1 + a))
    j = np.argmin(gam)
    gam_min = gam[j]
    b_ = np.copy(b)
    b_[is_active_] += gam_min * d
    lamb_ = lamb - gam_min
    return lamb, b, is_active, lamb_, b_, is_active_


def logit_lasso(x, y, pmax):
    # #####################################计算必要的常量################################ #
    # 变量个数p-1，样本量n
    n, p = np.shape(x)
    p += 1
    x = np.c_[np.ones(shape=(n, 1), dtype=float), x]
    xt = x.T
    y = y.astype("float")
    lamb, b, is_active, lamb_, b_, is_active_ = _logit_lasso_init(x, xt, y, n, p)
    df = 0
    while df < pmax:
        # 输入lamb_，初始值为 b_：
        # (1) 计算数值解b
        # (2) 如果活跃集元素个数小于pmax, 计算下一个lamb_和相应的初始值b_
        b, is_active, df, lamb_, b_, is_active_ = lars_iter(y, xt, b_, is_active_,
                                                                  lamb_, pmax)
        print("df= ", df)
        print(b[is_active])
    return b, is_active, lamb, df
