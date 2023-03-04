from jpeg2dct.numpy import loads
import numpy as np
from numba import jit
import sys

# yes, indexes list is repeated but numba complains otherwise

def hist_1D(dct, sf, his_range):
    return hist_2D(dct, sf, his_range).flatten()

@jit(nopython=True)
def hist_2D(dct, sf, his_range):
    sf_num = len(range(*sf))
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

    bin_num = len(range(*his_range)) + 1


    coords = indexes[sf[0]: sf[1]]

    # build histogram
    his = np.zeros((sf_num, bin_num))

    c_H = len(dct)
    c_W = len(dct[0])

    # # iterate over each 8x8 block in a patch and build up histogram
    for x in range(c_H):
        for y in range(c_W):
            sf = np.array([dct[x][y].reshape(8,8)[c[0]][c[1]] for c in coords])

            for i, f in enumerate(sf):
                h, _ = np.histogram(f, bins=bin_num, range=his_range)
                his[i] += h # update counts in histogram

    return his

@jit(nopython=True)
def his_encode(dct, sf_range, his_range):

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

    coords = indexes[sf_range[0]: sf_range[1]]


    c_H = len(dct)
    c_W = len(dct[0])

    his = np.zeros((20, 11, len(sf)))

    for x in range(c_H):
        for y in range(c_W):
            sf = np.array([dct[x][y].reshape(8,8)[c[0]][c[1]] for c in coords])

            for s in sf:
                for q in range(1, 21):

                    h, _ = np.histogram(s,bins=11, range=his_range)
                    h = h / q
                    his[q, :, s] += h
    
    return his

def process(patches, input):
    histograms = []

    for p in patches:
            # extract dct coefficients
            dct, _, _ =  loads(p)

            # this is just to stop numba complaining 
            his_range = (input.his_range[0], input.his_range[1])
            sf = (input.sf[0], input.sf[1])

            # build histograms
            histogram = getattr(sys.modules[__name__], input.hist_rep)(dct, sf, his_range)
            histograms.append(histogram)
            
    return histograms


   
            
