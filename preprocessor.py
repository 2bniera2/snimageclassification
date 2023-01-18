import generate_image_patches
import extract_dcts
import noise_extraction
import os
import random
import numpy as np
from sys import argv


task = {'train' : 0, 'val' : 1, 'test' : 2}

paths = [f"{os.getcwd()}/dataset/train", f"{os.getcwd()}/dataset/val", f"{os.getcwd()}/dataset/test"] 

outputs = [
    ('processed/X_train', 'processed/y_train'),
    ('processed/X_val', 'processed/y_val'),
    ('processed/X_test', 'processed/y_test')
]


path = paths[task[argv[1]]]
output = outputs[task[argv[1]]]

def chunk_size(x):
    return {
        '64': 64,
        'full': 0
    }.get(x, 64)

is_test = True if argv[1] == 'test' else False

#obtain training patches and labels
Examples, labels = generate_image_patches.generate_patches(path, chunk_size(argv[2]), is_test)

print(f"{len(Examples)} patches generated")



# preprocessing using original hyperparameters supplied by paper
X = extract_dcts.process(Examples, (0,9), (-50, 50))

y = np.array(labels)

np.save(f'{output[0]}_{argv[2]}.npy', X)
np.save(f'{output[1]}_{argv[2]}.npy', y)

### still need to do noise extraction








