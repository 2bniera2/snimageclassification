from jpeg2dct.numpy import loads
import numpy as np
import time as t
from numba import jit
import sys



@jit(nopython=True)
def hist_1D(dct, sf_range, his_range, bands, sf_num):
    bin_num = len(range(his_range[0], his_range[1])) + 1
    # indexes to DC and all AC coefficients
    indexes = [
        (0,0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2),(1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    # # obtain spatial frequencies
    if bands == 3:

        coords = [y for x in [indexes[sf_range[i][0]: sf_range[i][1]] for i in range(3)] for y in x]
    else:
        coords = indexes[sf_range[bands][0]: sf_range[bands][1]]

    # build histogram
    his = np.zeros((sf_num, bin_num))

    c_H = len(dct)
    c_W = len(dct[0])


    # # iterate over each 8x8 block in a patch and build up histogram
    for x in range(c_H):
        for y in range(c_W):
            sf = np.array([dct[x][y].reshape(8,8)[c[0]][c[1]] for c in coords])

            for i, f in enumerate(sf):
                h, _ = np.histogram(f, bins=bin_num, range=(his_range[0], his_range[1]))
                his[i] += h

    return his.flatten()


# def hist_2D

def process(patches, input):
    histograms = []
    
    for p in patches:
            dct, _, _ =  loads(p)
            
            # this is just to stop numba complaining 
            his_range = (input.his_range[0], input.his_range[1])
            sf_range = (
                (input.sf_range[0][0], input.sf_range[0][1]),
                (input.sf_range[1][0], input.sf_range[1][1]),
                (input.sf_range[2][0], input.sf_range[2][1])
            )
            # extract dct coefficients
            histogram = getattr(sys.modules[__name__], input.dct_rep)(dct, sf_range, his_range, input.bands,  input.sf_num)
            histograms.append(histogram)

    return histograms


   
            
