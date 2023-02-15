from jpeg2dct.numpy import loads
import numpy as np
import time as t
from numba import jit
import h5py



@jit(nopython=True)
def histogram_builder(dct, sf_range, his_range):
    bin_num = len(range(his_range[0], his_range[1])) + 1
    # indexes to all AC coefficients
    indexes = [
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5),
        (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5), (4, 6),
        (3, 7), (4, 7), (5, 6), (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    # # obtain spatial frequencies
    coords = indexes[0: sf_range]
    # build histogram
    his = np.zeros((sf_range,bin_num))

    c_H = len(dct)
    c_W = len(dct[0])

    # iterate over each 8x8 block in a patch and build up histogram
    for x in range(c_H):
        for y in range(c_W):
            sf = np.array([dct[x][y].reshape((8, 8))[c[0]][c[1]] for c in coords])

            for i, f in enumerate(sf):
                h, _ = np.histogram(f, bins=bin_num, range=(his_range[0], his_range[1]))
                his[i] += h

    return his.flatten()

def process(patches, sf_range, his_range):
    histograms = []
    
    for p in patches:
            # extract dct coefficients
            dct, _, _ =  loads(p, False)        
            histograms.append(histogram_builder(dct, sf_range, (his_range[0], his_range[1])))

    return histograms


   
            
