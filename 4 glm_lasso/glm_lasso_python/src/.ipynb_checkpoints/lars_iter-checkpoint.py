import numpy as np
from scipy import linalg
from src.downdating import downdating
from src.updating import updating


def _lars_init(w, xt, cc_t, is_active_t, active_set_t):
    cc_t_abs = np.absolute(cc_t)
    cci_t = np.argmax(cc_t_abs)
    cc_t_max = cc_t_abs[cci_t]
    lamb_t = cc_t_max
    is_active_t[cci_t] = True
    active_set_t.append(cci_t)
    xt_a = xt[is_active_t]
    xtx_a = np.dot(xt_a * w, xt_a.T)
    return lamb_t, linalg.cholesky(xtx_a, lower=True)


def _lars_step(xt, w, p, b_t, cc_t, active_set_t, is_active_t, left_mat_t, lamb_t, df_t):
    s_t = np.sign(cc_t)
    sa_t = s_t[active_set_t]
    sa_t[0] = 0
    d_t = linalg.cho_solve((left_mat_t, True), sa_t)
    u_t = np.dot(d_t, xt[active_set_t])
    a_t = np.dot(xt[~is_active_t], u_t * w)
    gam = np.ones(p, dtype=float) * lamb_t
    if df_t > 1:
        ww = - b_t[active_set_t] / d_t
        gam[active_set_t] = np.where(((ww > 0) & (ww < lamb_t)), ww, lamb_t)
        gam[0] = lamb_t
    if df_t < p - 1:
        gam[~is_active_t] = np.where(a_t * lamb_t <= cc_t[~is_active_t],
                                     (lamb_t - cc_t[~is_active_t]) / (1 - a_t),
                                     (lamb_t + cc_t[~is_active_t]) / (1 + a_t))
    return gam, d_t, sa_t, a_t


def lars_iter(y, xt, b_, is_active_, lamb, pmax):
    #  -----------------------------输入参数--------------------------------------------  #
    #  y: 因变量，一维数组, shape = (n, ); xt: 自变量, 二维数组, shape = (p, n)
    #  lamb: 调节参数
    #  b_ : 初始值
    #  is_active_: b_中非0元素位置
    #  pmax：模型中最大非0变量个数
    #  --------------------------------------------------------------------------------  #
    #
    b, is_active, df = None, None, None
    lamb_next, b_next, is_active_next = None, None, None
    p, n = np.shape(xt)
    count1 = 0
    while count1 < 100:
        count1 += 1
        eta = np.dot(b_[is_active_], xt[is_active_])
        exp_eta = np.exp(eta)
        mu = exp_eta / (1 + exp_eta)
        w = mu * (1 - mu)
        z = y - mu

        is_active_t = np.array([True] + [False] * (p - 1))
        active_set_t = [0]
        b_t = np.zeros(p, dtype=float)
        b_t[0] = np.sum(eta * w + z) / np.sum(w)
        cc_t = np.dot(xt, (eta * w) - (b_t[0] * w)) + np.dot(xt, z)
        lamb_t, left_mat_t = _lars_init(w, xt, cc_t, is_active_t, active_set_t)
        count2 = 0
        df_t = 1
        gam_min, gam_min_t, d_t = None, None, None
        while count2 < 2 * p:
            count2 += 1
            gam, d_t, sa_t, a_t = _lars_step(xt, w, p, b_t, cc_t,
                                             active_set_t, is_active_t,
                                             left_mat_t, lamb_t, df_t)
            j = np.argmin(gam)
            gam_min = gam[j]
            if lamb_t - gam_min < lamb:
                gam_min_t = lamb_t - lamb
            else:
                gam_min_t = gam_min
            b_t[active_set_t] += gam_min_t * d_t
            cc_t[active_set_t] -= gam_min_t * sa_t
            cc_t[~is_active_t] -= gam_min_t * a_t
            lamb_t = lamb_t - gam_min_t
            if lamb_t > lamb:
                if is_active_t[j]:
                    k = active_set_t.index(j)
                    _ = active_set_t.pop(k)
                    left_mat_t = downdating(left_mat_t, k)
                    df_t -= 1
                else:
                    xt_w = xt[j] * w
                    xtx_j = np.sum(xt[active_set_t] * xt_w, axis=1)
                    xtx_jj = np.sum(xt[j] * xt_w)
                    left_mat_t = updating(left_mat_t, xtx_j, xtx_jj)
                    active_set_t.append(j)
                    df_t += 1
                is_active_t[j] = ~ is_active_t[j]
            else:
                break
        eps = np.max(np.abs(b_t - b_))
        b_ = np.copy(b_t)
        # active_set_ = np.copy(active_set_t)
        is_active_ = np.copy(is_active_t)
        # is_active_next = is_active
        if eps < 1e-8:
            b = np.copy(b_)
            is_active = np.copy(is_active_)
            df = df_t
            is_active_next = np.copy(is_active)
            if df < pmax:
                b_next = np.copy(b_t)
                if gam_min_t != gam_min:
                    lamb_next = lamb - (gam_min - gam_min_t)
                    b_next[active_set_t] += (gam_min - gam_min_t) * d_t
                else:
                    gam, d_t, _, _ = _lars_step(xt, w, p, b_t, cc_t,
                                                active_set_t, is_active_t,
                                                left_mat_t, lamb_t, df_t)
                    j = np.argmin(gam)
                    gam_min = gam[j]
                    lamb_next = lamb - gam_min
                    b_next[active_set_t] += gam_min * d_t
                if lamb - lamb_next < 0.01:
                    lamb_next = lamb * 0.95
            break
    return b, is_active, df, lamb_next, b_next, is_active_next
