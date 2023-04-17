from jpeg2dct.numpy import load, loads
import numpy as np
import os
from numba import jit
import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

patch_size = 0
his_range = [-100, 100]
sf_range = [1, 63]
patch_counter = 0


class_dict = {
    ['facebook': np.zeros((len(range(*sf_range)), len(range(*his_range)) + 1))],
    ['instagram':np.zeros((len(range(*sf_range)), len(range(*his_range)) + 1))],
    ['orig':np.zeros((len(range(*sf_range)), len(range(*his_range)) + 1))],
    ['telegram':np.zeros((len(range(*sf_range)), len(range(*his_range)) + 1))],
    ['twitter':np.zeros((len(range(*sf_range)), len(range(*his_range)) + 1))],
    ['whatsapp':np.zeros((len(range(*sf_range)), len(range(*his_range)) + 1))]
}

@jit(nopython=True)
def hist(dct, sf, his_range):
    sf_num = len(range(*sf))

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


def im_to_bytes(patch, q):
    buf = io.BytesIO()
    patch.save(buf, format='JPEG', qtables=q)
    return buf.getvalue()

def make_patches(image, patch_size, q=None, to_bytes=True):
    patches = []
    for i in range(0, image.width-patch_size+1, patch_size):
        for j in range(0, image.height-patch_size+1, patch_size):
            patch = image.crop((i, j, i+patch_size, j+patch_size))
            if to_bytes:
                patch = im_to_bytes(patch, q)
            patches.append(patch)
    return patches  


# iterate over each image per class and 
for CLASS in os.listdir(f'{os.getcwd()}/sample'):
    patch_counter = 0
    for IMAGE in os.listdir(f'{os.getcwd()}/sample/{CLASS}'):
        path = f'{os.getcwd()}/sample/{CLASS}/{IMAGE}'

        image = Image.open(path)
        qtable = image.quantization

        his_range = (his_range[0], his_range[1])
        sf = (sf_range[0], sf_range[1])

        if patch_size:
            patches = make_patches(image, patch_size, qtable, True)
            for patch in patches:
                dct, _, _ = loads(patch)
                histogram = hist(dct, sf, his_range)
                class_dict[CLASS] += histogram
        else:
            dct, _, _ = load(path) 
            histogram = hist(dct, sf, his_range)
            class_dict[CLASS] += histogram
       
for class_hist in class_dict.items():
    np.save(f'{os.getcwd()}/histogram_processed/{class_hist[0]}_{patch_size}_avg', class_hist[1])


