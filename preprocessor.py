'''
run program using:
python preprocessor.py {task} {chunk_size} {name}

{task} = 'train' | 'val' | 'val'

{chunk_size} = '64' | 'full'

{name} = short name for output file
'''


import generate_image_patches as generate_image_patches
import extract_dcts as extract_dcts
import noise_extraction as noise_extraction
import os
import numpy as np
from sys import argv


task = {'train' : 0, 'val' : 1, 'test' : 2}

path = f"{os.getcwd()}/dataset"

outputs = [
    ('processed/X_train', 'processed/y_train'),
    ('processed/X_val', 'processed/y_val'),
    ('processed/X_test', 'processed/y_test')
]


output = outputs[task[argv[1]]]

def chunk_size(x):
    return {
        '64': 64,
        'full': 0
    }.get(x, 64)

is_test = True if argv[1] == 'test' else False

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