import numpy as np


def scale(x, is_scale=True):
    x_bar = np.mean(x, axis=0)
    x1 = np.copy(x)
    x1 = x1 - x_bar
    if is_scale:
        x_std = np.std(x, axis=0, ddof=1)
        x1 = x1 / x_std
        return x1, x_bar, x_std
    return x1, x_bar
