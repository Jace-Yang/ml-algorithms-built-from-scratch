import numpy as np
from scipy import linalg
from scipy.stats import norm
from src.sigma_ma import sigma_ma


def sim(n, p, rho, mu, beta0, beta1):
    var = sigma_ma(p, rho)
    uper_mat = linalg.cholesky(var)
    x = norm.rvs(size=(n, p))
    x = np.dot(x, uper_mat)
    x = x + mu
    y = beta0 + np.dot(x, beta1) + norm.rvs(size=n)
    return x, y
