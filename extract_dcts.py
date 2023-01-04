from jpeg2dct.numpy import load, loads
import numpy as np
from typing import Tuple, List
from itertools import product
import sys

np.set_printoptions(threshold=sys.maxsize)

def process(patches: List[Tuple[str, bytes]], sf_range: Tuple[int, int], histogram_range: Tuple[int, int]):
    X = []

    # indexes to all AC coefficients
    indexes = [
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5),
        (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5), (4, 6),
        (3, 7), (4, 7), (5, 6), (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    
    for p in patches:
        # extract dct coefficients
        dct, _, _ = loads(p)
        print(dct[0][0])

        # obtain spatial frequencies
        coords = indexes[sf_range[0]: sf_range[1]]

        # build histogram
        his = np.zeros((len(coords), len(range(*(histogram_range))) + 1))

        c_H = len(dct)
        c_W = len(dct[0])

        # iterate over each 8x8 block in a patch and build up histogram
        for x, y in product(range(c_H), range(c_W)): 
            sf = np.array([dct[x][y][c[0]][c[1]]] for c in coords)

            for i, f in enumerate(sf):
                h, b = np.histogram(f, bins=len(range(*(histogram_range))) + 1, range=histogram_range)
                his[i] += h

        
        X.append(his.flatten())

    return np.array(X)





            




        
