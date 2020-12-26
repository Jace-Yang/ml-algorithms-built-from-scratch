import numpy as np
import math

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
