import numpy as np
from patchify import patchify
import cv2

def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()



def make_patches(images, labels, size, test):
    X_b = []
    y = []


    for index, (image, label) in enumerate(zip(images, labels)):
        patches = patchify(image, (size, size, 3), step=size)

        #iterate over all patches
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                
                X_b.append(im_to_bytes(patch))
                y.append(f"{label}.{index}" if test else label)
    
    return X_b, y
                
      
        






