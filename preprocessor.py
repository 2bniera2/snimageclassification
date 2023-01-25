'''

'''
import load_images as load_images
import make_patches as make_patches
import extract_dcts as extract_dcts
import extract_noise as extract_noise
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sys import argv


path = f"{os.getcwd()}/dataset"

# load iamages
images, labels = load_images.load_images(path)


#split data
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=1)

images_test, labels_test, images_val, labels_val = train_test_split(
    images_test, labels_test, test_size = 0.5, random_state=1)

# make patches
train_patches_bytes, train_patches, labels_train = make_patches()
val_patches_bytes, val_patches, labels_val =
test_patches_bytes, test_patches, labels_val = 


# preprocess 



X_train = X = extract_dcts.process(, (0,9), (-50, 50))
X_val = X = extract_dcts.process(Examples_bytes, (0,9), (-50, 50))
X_test = X = extract_dcts.process(Examples_bytes, (0,9), (-50, 50))





#obtain training patches and labels
Examples_bytes, Examples, labels = generate_image_patches.generate_patches(path, chunk_size(argv[2]), is_test)

print(f"{len(Examples)} patches generated")



# # preprocessing using original hyperparameters supplied by paper
X = extract_dcts.process(Examples_bytes, (0,9), (-50, 50))

y = np.array(labels)

np.save(f'{output[0]}_{argv[3]}.npy', X)
np.save(f'{output[1]}_{argv[3]}.npy', y)

## still need to do noise extraction


# X = noise_extraction.extract(Examples)






# X = noise_extraction.extract(Examples)
# y = np.array(labels)

# np.save(f'{output[0]}_noise.npy', X)
# np.save(f'{output[1]}_noise.npy', y)