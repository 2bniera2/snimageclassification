from jpeg2dct.numpy import loads
import numpy as np
import time as t
from numba import jit
import h5py



@jit(nopython=True)
def histogram_builder(dct, sf_range, his_range):
    # indexes to all AC coefficients
    indexes = [
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5),
        (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5), (4, 6),
        (3, 7), (4, 7), (5, 6), (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    # obtain spatial frequencies
    coords = indexes[sf_range[0]: sf_range[1]]

    # build histogram
    his = np.zeros((len(coords), len(range(*(his_range))) + 1))

    c_H = len(dct)
    c_W = len(dct[0])

    # iterate over each 8x8 block in a patch and build up histogram
    for x in range(c_H):
        for y in range(c_W):
            sf = np.array([dct[x][y].reshape((8, 8)).T[c[0]][c[1]]
                           for c in coords])

            for i, f in enumerate(sf):
                h, b = np.histogram(f, bins=len(
                    range(*(his_range))) + 1, range=his_range)
                his[i] += h

    return his.flatten()

def process(patches, sf_range, his_range, task, name):
    with h5py.File(f'processed/DCT_{task}_{name}.h5', 'w') as f:
        dset = f.create_dataset('DCT', shape=(0, 909), maxshape=(None, 909))

  

    
    for p in patches:
        with h5py.File(f'processed/DCT_{task}_{name}.h5', 'a') as f:
            # extract dct coefficients
            dct, _, _ =  loads(p, False)        
            dset = f['DCT']
            dset.resize((dset.shape[0] + 1, 909))
            dset[-1] = histogram_builder(dct, sf_range, his_range)
            