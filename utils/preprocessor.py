'''
Usage:
python preprocessor.py {size} {name}
'''
from utils.load_images import load_images
from utils.make_patches import builder
from utils.extract_dcts import process
from sklearn.model_selection import train_test_split
import time as time
import os


def main(patch_size, name, his_range, sf_range):
    path = f'{os.getcwd()}/dataset'

    t1 = time.time()
    # load images
    images, labels = load_images(path)
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
    train_patches = builder(
        images_train, labels_train, patch_size, 'train', name)
    val_patches = builder(
        images_val, labels_val, patch_size, 'val', name)
    test_patches = builder(
        images_test, labels_test, patch_size, 'test', name)

    t6 = time.time()

    print(f'Making patches time: {t6 - t5}')

    t7 = time.time()
    # # dct
    process(train_patches, sf_range, his_range, 'train', name)
    process(val_patches, sf_range, his_range, 'val', name)
    process(test_patches, sf_range, his_range, 'test', name)

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
