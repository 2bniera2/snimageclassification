import numpy as np
import cv2

def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()

def make_patches(im, size):
    patches = []
    for i in range(0, im.shape[0]-size+1, size):
        for j in range(0, im.shape[1]-size+1, size):
            patches.append(im[i:i+size, j:j+size])
    return patches

def builder(paths, labels, size, test):
    X_b = []
    y = []


    for index, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        patches = make_patches(image, size)

        #iterate over all patches
        for patch in patches:
            X_b.append(im_to_bytes(patch))
            y.append(f"{label}.{index}" if test else label)
    
    return X_b, y
                
      
        






