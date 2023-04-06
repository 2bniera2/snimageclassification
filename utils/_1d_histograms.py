from jpeg2dct.numpy import loads, load
import numpy as np
from numba import jit

@jit(nopython=True)
def hist(dct, sf, his_range):
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

    return his.flatten()

# input a list of patches 
# output a list of histograms
def process_patches(patches, input):
    histograms = []

    for p in patches:
            # extract dct coefficients
            dct, _, _ =  loads(p)

            # this is just to stop numba complaining 
            his_range = (input.his_range[0], input.his_range[1])
            sf = (input.sf[0], input.sf[1])

            # build histograms
            histogram = hist(dct, sf, his_range)
            histograms.append(histogram)
            
    return histograms

# input a image path
# output a single histogram
def process(image, input):
    dct, _, _ = load(image)

    # this is just to stop numba complaining 
    his_range = (input.his_range[0], input.his_range[1])
    sf = (input.sf[0], input.sf[1])

    return hist(dct, sf, his_range)


            
