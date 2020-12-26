import numpy as np
from src.sim_data import sim_data
from src.logit_lasso import logit_lasso


np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
family = "logit"
rho = 0.5
n, p, s = 400, 1000, 6
pmax = 10
beta_0 = np.array([0.05, 3, -2.5, 3.5, -1.5, -3])
beta_1 = np.zeros(p - s)
beta = np.append(beta_0, beta_1)
x, y = sim_data(beta, rho, n, family)
b, is_active, lamb, df = logit_lasso(x, y, pmax=10)
print(b[:10])
print(b[is_active])
print(lamb)
