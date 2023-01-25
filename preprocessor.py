'''
Usage:
python preprocessor.py {size} {name}
'''
import utilities.load_images as load_images
import utilities.make_patches as make_patches
import utilities.extract_dcts as extract_dcts
import utilities.extract_noise as extract_noise
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

# load images
images, labels = load_images.load_images(path)

#split data
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, shuffle=False)

images_val, images_test, labels_val, labels_test = train_test_split(
    images_test, labels_test, test_size = 0.5, shuffle=False)





# make patches
train_patches_bytes, y_train = make_patches.make_patches(images_train, labels_train, size, False)
val_patches_bytes, y_val = make_patches.make_patches(images_val, labels_val, size, False)
test_patches_bytes, y_test = make_patches.make_patches(images_test, labels_test, size, True)

print("patches made")

## preprocess 

# dct
X_train = extract_dcts.process(train_patches_bytes, (0,9), (-50, 50))
X_val = extract_dcts.process(val_patches_bytes, (0,9), (-50, 50))
X_test = extract_dcts.process(test_patches_bytes, (0,9), (-50, 50))

print("dct extracted")

# # noise
# Xn_train = extract_noise.extract(images, size)
# Xn_val = extract_noise.extract(images, size)
# Xn_test = extract_noise.extract(images, size)

print("noise residuals extracted")

# save dct
np.save(f"processed/X_DCT_train_{argv[2]}.npy", X_train)
np.save(f"processed/X_DCT_val_{argv[2]}.npy", X_val)
np.save(f"processed/X_DCT_test_{argv[2]}.npy", X_test)
# #save noise
# np.save(f"processed/X_noise_train_{argv[2]}.npy", X_train)
# np.save(f"processed/X_noise_val_{argv[2]}.npy", X_val)
# np.save(f"processed/X_noise_test_{argv[2]}.npy", X_test)

#save labels
np.save(f"processed/y_train_{argv[2]}.npy", y_train)
np.save(f"processed/y_val_{argv[2]}.npy", y_val)
np.save(f"processed/y_test_{argv[2]}.npy", y_test)


