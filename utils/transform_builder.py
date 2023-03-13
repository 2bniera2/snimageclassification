import cv2
import numpy as np
import pywt
import h5py
from numba import jit
from jpeg2dct.numpy import load


def to_dct_domain(path, input_shape):
    image = cv2.imread(path, 0)
    image = cv2.dct(np.float32(image))
    image = cv2.resize(image, input_shape)
    dct = np.stack((image, image, image)).reshape((*input_shape, 3))
    return dct

    
@jit(nopython=True)
def histogram_extract(dct):
    sf = [1, 63]
    his_range = (-100, 100)
    sf_num = len(range(1, 63))
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


# def to_dct_domain(path, input_shape):
#     dct, _, _ = load(path)
#     his =  histogram_extract(dct)
#     his = np.stack((his, his, his)).reshape((*input_shape, 3))

#     return his


def to_dwt_domain(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dwt = pywt.wavedec2(im, 'bior4.4', level=2)
    dwt, _ = pywt.coeffs_to_array(dwt)
    dwt = cv2.resize(dwt, (224, 224))
    dwt = np.repeat(dwt[:,:,np.newaxis], 3, axis=-1)

    return dwt



def transform_builder(input, task, examples, labels):
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, *input.input_shape, 3), maxshape=(None, *input.input_shape, 3))
        _ = f.create_dataset('labels', shape=(0, 1), maxshape=(None, 1))


        for im_num, (path, label) in enumerate(zip(examples, labels)):
            print(f'{im_num+1}/{len(examples)}')
            
            if input.domain == "DCT":
                im = to_dct_domain(path, input.input_shape)

            elif input.domain == "DWT":
                im = to_dwt_domain(path)


            dct_dset = f['examples']
            dct_dset.resize((dct_dset.shape[0] + 1, *input.input_shape, 3))
            dct_dset[-1] = im
                    
            labels_dset = f['labels']
            labels_dset.resize((labels_dset.shape[0] + 1, 1))
            labels_dset[-1] = np.array([label])