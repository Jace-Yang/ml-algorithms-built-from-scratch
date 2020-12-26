import numpy as np
import math

def sigmoidThreshold(Xb, thetas, threshold = 0.5):
    params = - Xb.dot(thetas)
    outcome = np.zeros(params.shape[0])
    for i in range(len(outcome)):
        outcome[i] = 1 /(1 + math.exp(params[i]))
        if outcome[i] >= threshold:
            outcome[i] = 1
        else:
            outcome[i] = 0
    return outcome
