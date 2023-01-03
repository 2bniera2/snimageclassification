import generate_image_patches
import extract_dcts
import os
import random

TRAIN = 0
VALIDATE = 1
TEST = 2


train_path = f"{os.getcwd()}\\dataset_1\\train"

val_path = f"{os.getcwd()}\\dataset_1\\val"

#obtain training patches and labels
X_train, y = generate_image_patches.generate_patches(train_path, 64, TRAIN)

X_train = extract_dcts.process()





