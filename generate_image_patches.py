import numpy as np
from patchify import patchify
from PIL import Image
import os
import io
import random


def shuffle_data(patches, labels): 
    temp = list(zip(patches, labels))
    random.shuffle(temp)
    patches, labels = zip(*temp) 
    return patches, labels

def im_to_bytes(patch):
    patch = Image.fromarray(patch)
    patch_bytes = io.BytesIO()
    patch.save(patch_bytes, format="JPEG")
    return patch_bytes.getvalue()



def generate_patches(path: str, patch_size: int, test: bool):
    X = []
    labels = []
    for class_name in os.listdir(path):
        # list of files in that class
        file_list = os.listdir(f"{path}/{class_name}")
        
        # patchify each file in the class 
        for index, file in enumerate(file_list):
            im = np.asarray(Image.open(f"{path}/{class_name}/{file}", 'r'))
            patches = patchify(im, (patch_size, patch_size, 3), step=patch_size)

            #convert patches to byte string and append to list along with label
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0]
                    num = i * patches.shape[1] + j

                    X.append(im_to_bytes(patch))


                    labels.append(f"{class_name}") if not test else labels.append(f"{class_name}.{index}.{num}")
    
    return shuffle_data(X, labels)

            



