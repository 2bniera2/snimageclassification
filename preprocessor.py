import generate_image_patches
import extract_dcts
import os
import random
import numpy as np


train_path = f"{os.getcwd()}/dataset_1/train"

val_path = f"{os.getcwd()}/dataset_1/val"

#obtain training patches and labels
Examples, labels = generate_image_patches.generate_patches(train_path, 64, False)




print(f"{len(Examples)} patches generated")

# preprocessing using original hyperparameters supplied by paper
processed_X_train = extract_dcts.process(Examples, (0,9), (-50, 50))
y_array = np.array(labels)

# # try first half of block
# processed_X_train = extract_dcts.process(X_train, (0,35), (-50, 50))

# # try second half of block
# processed_X_train = extract_dcts.process(X_train, (27,63), (-50, 50))

# # try all of block
# processed_X_train = extract_dcts.process(X_train, (0, 63), (-50, 50))

np.save('data.npy', processed_X_train)
np.save('labels.npy', y_array)










