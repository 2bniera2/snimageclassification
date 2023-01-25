'''
Usage:
python preprocessor.py {size} {name}
'''
import load_images as load_images
import make_patches as make_patches
import extract_dcts as extract_dcts
import extract_noise as extract_noise
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sys import argv


match argv[1]:
    case "full": 
        size = -1
    case _:
        size = 64
        


path = f"{os.getcwd()}/dataset"

# load iamages
images, labels = load_images.load_images(path)


#split data
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=1)

images_test, labels_test, images_val, labels_val = train_test_split(
    images_test, labels_test, test_size = 0.5, random_state=1)

# make patches
train_patches_bytes, train_patches, y_train = make_patches.make_patches(images_train, labels_train, size, False)
val_patches_bytes, val_patches, y_val = make_patches.make_patches(images_val, labels_val, size, False)
test_patches_bytes, test_patches, y_test = make_patches.make_patches(images_test, labels_test, size, True)


# preprocess 

# dct
X_train = extract_dcts.process(train_patches_bytes, (0,9), (-50, 50))
X_val = extract_dcts.process(val_patches_bytes, (0,9), (-50, 50))
X_test = extract_dcts.process(test_patches_bytes, (0,9), (-50, 50))

# noise
Xn_train = extract_noise.extract(train_patches)
Xn_val = extract_noise.extract(val_patches)
Xn_test = extract_noise.extract(test_patches)


# save dct
np.save(f"processed/X_DCT_train_{argv[2]}.npy", X_train)
np.save(f"processed/X_DCT_val_{argv[2]}.npy", X_val)
np.save(f"processed/X_DCT_test_{argv[2]}.npy", X_test)
#save noise
np.save(f"processed/X_noise_train_{argv[2]}.npy", X_train)
np.save(f"processed/X_noise_val_{argv[2]}.npy", X_val)
np.save(f"processed/X_noise_test_{argv[2]}.npy", X_test)

#save labels
np.save(f"processed/y_train_{argv[2]}.npy", y_train)
np.save(f"processed/y_val_{argv[2]}.npy", y_val)
np.save(f"processed/y_test_{argv[2]}.npy", y_test)


