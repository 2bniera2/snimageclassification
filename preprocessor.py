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
import time as time


def main():

    # user defined variables
    size = -1 if argv[1] == 'full' else 64
    path = f"{os.getcwd()}/dataset"
    name = argv[2]

    t1 = time.time()
    # load images
    images, labels = load_images.load_images(path)
    t2 = time.time()

    print(f'Load image time: {t2 - t1}')

    t3 = time.time()

    # split data
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)

    images_val, images_test, labels_val, labels_test = train_test_split(
        images_test, labels_test, test_size=0.5, random_state=42, stratify=labels_test)

    t4 = time.time()

    print(f'Split time: {t4 - t3}')

    t5 = time.time()
    # # make patches
    train_patches = make_patches.builder(
        images_train, labels_train, size, 'train', name)
    val_patches = make_patches.builder(
        images_val, labels_val, size, 'val', name)
    test_patches = make_patches.builder(
        images_test, labels_test, size, 'test', name)

    t6 = time.time()

    print(f'Making patches time: {t6 - t5}')

    t7 = time.time()
    # # dct
    extract_dcts.process(train_patches, (0, 9), (-50, 50), 'train', name)
    extract_dcts.process(val_patches, (0, 9), (-50, 50), 'val', name)
    extract_dcts.process(test_patches, (0, 9), (-50, 50), 'test', name)

    t8 = time.time()

    print(f'DCT extraction time: {t8 - t7}')

    print(f'Total time taken {(t8-t7)+(t6-t5)+(t4-t3)+(t2-t1)}')

if __name__ == "__main__":
    main()



# print("Noise extraction start")

# # extract noise as well as divide image into patches and then save
# extract_noise.extract(images_train, size, 'train', name)
# extract_noise.extract(images_val, size, 'val', name)
# extract_noise.extract(images_test, size, 'test', name)

# print("Finished noise extraction")

# print("Finished Preprocessing")


# print('Saving labels')

# # #save labels, will be for both dcts and noise
# np.save(f"processed/y_train_{name}.npy", y_train)
# np.save(f"processed/y_val_{name}.npy", y_val)
# np.save(f"processed/y_test_{name}.npy", y_test)

# print('Finished saving labels')


# # # save dct
# np.save(f"processed/X_train_{name}.npy", X_train)
# np.save(f"processed/X_val_{name}.npy", X_val)
# np.save(f"processed/X_test_{name}.npy", X_test)
