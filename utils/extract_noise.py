import cv2
from PIL import Image
import h5py
import numpy as np
import utilities.make_patches as make_patches
import time 

def extract_noise_residual(i, d):
    
    n_r = i - d
    n_r = (n_r - np.min(n_r)) / (np.max(n_r) - np.min(n_r))

    return n_r


def extract(paths, size, task, dataset_name):
    with h5py.File(f'processed/noise_{task}_{dataset_name}.h5', 'w') as f:
        dset = f.create_dataset('noise', shape=(0, size, size), maxshape=(None, size, size))

    for p in paths:
        with h5py.File(f'processed/noise_{task}_{dataset_name}.h5', 'a') as f:

            i = np.array(Image.open(p).convert('L'))

            d = cv2.fastNlMeansDenoising(i)

            residual = extract_noise_residual(i, d)
            patches = make_patches.make_patches(residual, size)

            dset = f['noise']

            dset.resize((dset.shape[0] + len(patches), size, size))

            dset[-len(patches):] = patches 