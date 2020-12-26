import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
