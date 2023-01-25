import cv2
import numpy as np
from numba import jit
import time as time

@jit(nopython=True)
def extract_noise_residual(i, d):
    
    n_r = i - d
    n_r = (n_r - np.min(n_r)) / (np.max(n_r) - np.min(n_r))

    return n_r

@jit(nopython=True)
def make_patches(residual, size):
    patches = []
    for i in range(0, residual.shape[0]-size[0]+1, size[0]):
        for j in range(0, residual.shape[1]-size[1]+1, size[1]):
            patch = residual[i:i+size[0], j:j+size[1]]
            patches.append(patch)
    return patches

def extract(images, size):
    
    X = []
    size = (size, size)

    for i in images:
        start = time.time()
        d = cv2.fastNlMeansDenoisingColored(i)

        residual = extract_noise_residual(i, d)
        patches = make_patches(residual, size)
        print(f'{time.time() - start}')
        X.extend(patches)


    return X


