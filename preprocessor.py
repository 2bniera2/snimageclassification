import generate_image_patches
import extract_dcts
import os
import random
import numpy as np
import sys


task = {'train' : 0, 'val' : 1, 'test' : 2}

paths = [f"{os.getcwd()}/dataset_1/train", f"{os.getcwd()}/dataset_1/val", f"{os.getcwd()}/dataset_1/test"] 
outputs = [('X_train.npy', 'y_train.npy'), ('X_val.npy', 'y_val.npy'), ('X_test.npy', 'y_test.npy')]


path = paths[task[sys.argv[1]]]
output = outputs[task[sys.argv[1]]]

def chunk_size(x):
    return {
        '8' : 8,
        '64': 64,
        'full': 0
    }.get(x, 64)

is_test = True if sys.argv[1] == 'test' else False

#obtain training patches and labels
Examples, labels = generate_image_patches.generate_patches(path, chunk_size(sys.argv[2]), is_test)

print(f"{len(Examples)} patches generated")

# preprocessing using original hyperparameters supplied by paper
X = extract_dcts.process(Examples, (0,9), (-50, 50))
y = np.array(labels)

np.save(output[0], X)
np.save(output[1], y)










