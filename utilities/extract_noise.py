import cv2
from PIL import Image
import numpy as np
import time 

def extract_noise_residual(i, d):
    
    n_r = i - d
    n_r = (n_r - np.min(n_r)) / (np.max(n_r) - np.min(n_r))

    return n_r

def make_patches(residual, size):
    patches = []
    for i in range(0, residual.shape[0]-size[0]+1, size[0]):
        for j in range(0, residual.shape[1]-size[1]+1, size[1]):
            patch = residual[i:i+size[0], j:j+size[1]]
            patches.append(patch)
    return patches

def extract(paths, size):
    
    X = []
    size = (size, size)

    for p in paths:
        i = np.array(Image.open(p).convert('L'))

        d = cv2.fastNlMeansDenoising(i)

        residual = extract_noise_residual(i, d)
        patches = make_patches(residual, size)
        X.extend(patches)


    return np.array(X)


