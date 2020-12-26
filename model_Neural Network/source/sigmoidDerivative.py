import numpy as np
import math
from source.sigmoid import sigmoid

def sigmoidDerivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)
