import numpy as np
def scale(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0, ddof=1)  # ddof: degrees of freedom, default 0
    x_scaled = np.copy(x)  # don't change original x
    x_scaled = (x - x_mean)/x_std
    return x_scaled, x_mean, x_std
    