from skimage.restoration import denoise_wavelet
import numpy as np
import time as time


def extract_noise_residual(image) -> np.ndarray:
    denoised_image = denoise_wavelet(image)
    n_r = image - denoised_image
    n_r = (n_r - np.min(n_r)) / (np.max(n_r) - np.min(n_r))

    return n_r


def extract(patches):
    
    X = np.empty((len(patches), 64, 64, 3))
    i = 0


    for p in patches:

        
        n_r = extract_noise_residual(p)
        X[i] = n_r
        i+=1

    return X


