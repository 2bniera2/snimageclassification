import generate_image_patches
import extract_dcts
import os
import random

train_path = f"{os.getcwd()}/dataset_1/train"

val_path = f"{os.getcwd()}/dataset_1/val"

#obtain training patches and labels
X_train, y = generate_image_patches.generate_patches(train_path, 64, False)

print(f"{len(X_train)} patches generated")


# preprocessing using original hyperparameters supplied by paper
processed_X_train = extract_dcts.process(X_train, (0,9), (-50, 50))

print("X processed")

# # try first half of block
# processed_X_train = extract_dcts.process(X_train, (0,35), (-50, 50))

# # try second half of block
# processed_X_train = extract_dcts.process(X_train, (27,63), (-50, 50))

# # try all of block
# processed_X_train = extract_dcts.process(X_train, (0, 63), (-50, 50))









