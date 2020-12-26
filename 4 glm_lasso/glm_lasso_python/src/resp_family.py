import numpy as np
from scipy.stats import norm, bernoulli, poisson


def _resp_logit(x, beta):
    eta = np.dot(x, beta)
    expeta = np.exp(eta)
    mu = expeta / (1 + expeta)
    return bernoulli.rvs(p=mu)


def _resp_gaussian(x, beta):
    mu = np.dot(x, beta)
    return norm.rvs(loc=mu)


def _resp_poisson(x, beta):
    eta = np.dot(x, beta)
    mu = np.exp(eta)
    return poisson.rvs(mu)

resp_family = {
    "logit": _resp_logit,
    "probit": _resp_gaussian,
    "poisson": _resp_poisson,
 }
