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


def extract(input, task, examples, labels):
    with h5py.File(f'processed/noise_{task}_{input.noise_dset_name}.h5', 'w') as f:
        _ = f.create_dataset('noise', shape=(0, input.patch_size, input.patch_size), maxshape=(None, input.patch_size, input.patch_size))
        _ = f.create_dataset('labels', shape=(0, 2), maxshape=(None, 2))


        for im_num, (path, label) in enumerate(zip(examples, labels)):

                i = np.array(Image.open(path).convert('L'))

                d = cv2.fastNlMeansDenoising(i)

                residual = extract_noise_residual(i, d)
                patches = make_patches.make_patches(residual, input.patch_size)

                for patch in patches:


                    dset = f['noise']
                    dset.resize((dset.shape[0] + 1, input.patch_size, input.patch_size))
                    dset[-1] = patch 

                    dset = f['labels']
                    dset.resize((dset.shape[0] + 1, 2))
                    dset[-1] = np.array([label, im_num])