import numpy as np
from patchify import patchify
from PIL import Image
import os
import io
import random
import cv2




def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()

def generate_patches(path: str, patch_size: int, test: bool):
    X = []
    labels = []

    for class_name in os.listdir(path):
        # list of files in that class
        file_list = os.listdir(f"{path}/{class_name}")
        
        # patchify each image in the class 
        for index, file in enumerate(file_list):
            print(f"class: {class_name} image_no.: {index}")
            if patch_size != 0:
                im = np.asarray(Image.open(f"{path}/{class_name}/{file}", 'r'))
                patches = patchify(im, (patch_size, patch_size, 3), step=patch_size)

                #convert patches to byte string and append to list along with label
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, 0]

                        X.append(im_to_bytes(patch))
                        labels.append(f"{class_name}") if not test else labels.append(f"{class_name}.{index}")
            else:
                with open(f"{path}/{class_name}/{file}", 'rb') as src:  buffer = src.read()
                X.append(buffer)
                labels.append(f"{class_name}") 
                

    return X, labels
            



