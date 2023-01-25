import numpy as np
from patchify import patchify
import cv2

def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()



def generate_patches(images, labels, patch_size, test):
    n = len(images)
    X_b = np.empty(n)
    X_p = np.empty((n, patch_size, patch_size, 3))
    y = np.empty(n)
    i = 0

    for index, image, label in enumerate(zip(images, labels)):
        patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)

        #iterate over all patches
        for x in range(patches.shape[0]):
            for y in range(patches.shape[1]):
                patch = patches[x, y, 0]
                
                X_b[i] = im_to_bytes(patch)
                X_p[i] = patch
                y[i] = f"{label}.{index}" if test else label

                i+=1

    return X_b, X_p, y
                
      
        






