from scipy.stats import norm
from scipy import linalg as sl
import numpy as np
from src.sigma_mat import sigma_ma
from src.resp_family import resp_family


def sim_data(beta, rho, n, family):
    p = np.shape(beta)[0]
    cov = sigma_ma(p - 1, rho)
    uper_mat = sl.cholesky(cov)
    x = norm.rvs(size=(n, p - 1))
    x = np.dot(x, uper_mat)
    x_1 = np.c_[np.ones(shape=(n, 1), dtype=float), x]
    y = resp_family.get(family)(x_1, beta)
    return x, y
