import cv2
import numpy as np
from numba import jit
import time as time

@jit(nopython=true)
def extract_noise_residual(i, d):
    
    n_r = i - d
    n_r = (n_r - np.min(n_r)) / (np.max(n_r) - np.min(n_r))

    return n_r


def extract(patches):
    
    X = np.empty((len(patches), 64, 64, 3))
    i = 0

    for p in patches:
        d = fastNlMeansDenoisingColoured(p)
        X[i] = extract_noise_residual(p, d)
        i+=1

    return X


