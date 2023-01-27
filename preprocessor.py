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

        
size = -1 if argv[1] == 'full' else 64

path = f"{os.getcwd()}/dataset"

# load images
images, labels = load_images.load_images(path)

#split data
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=1, stratify=labels)

images_val, images_test, labels_val, labels_test = train_test_split(
    images_test, labels_test, test_size = 0.5, random_state=1, stratify=labels_test)





# # make patches
# train_patches, y_train = make_patches.builder(images_train, labels_train, size, False)
# val_patches, y_val = make_patches.builder(images_val, labels_val, size, False)
# test_patches, y_test = make_patches.builder(images_test, labels_test, size, True)

# print("patches made")

# ## preprocess 
# print("DCT extraction start")
# # dct
# X_train = extract_dcts.process(train_patches, (0,9), (-50, 50))
# X_val = extract_dcts.process(val_patches, (0,9), (-50, 50))
# X_test = extract_dcts.process(test_patches, (0,9), (-50, 50))

# # save dct
# np.save(f"processed/X_train_{argv[2]}.npy", X_train)
# np.save(f"processed/X_val_{argv[2]}.npy", X_val)
# np.save(f"processed/X_test_{argv[2]}.npy", X_test)

# #save labels
# np.save(f"processed/y_train_{argv[2]}.npy", y_train)
# np.save(f"processed/y_val_{argv[2]}.npy", y_val)
# np.save(f"processed/y_test_{argv[2]}.npy", y_test)

# print("DCTs labels saved")



print("Noise extraction start")

N_train = extract_noise.extract(images_train, size)
N_val = extract_noise.extract(images_val, size)
N_test= extract_noise.extract(images_test, size)


np.save(f"processed/Noise_train_{argv[2]}.npy", N_train)
np.save(f"processed/Noise_val_{argv[2]}.npy", N_val)
np.save(f"processed/Noise_test_{argv[2]}.npy", N_test)

print("Noise extraction finished and saved")
