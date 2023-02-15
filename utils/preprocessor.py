'''
Usage:
python preprocessor.py {size} {name}
'''
from utils.load_images import load_images
from utils.make_patches import builder
from utils.extract_dcts import process
import time as time
import os


def main(patch_size, dataset_name, his_range, sf_range):
    path = f'{os.getcwd()}'

   
    # load images and split data
    images_train, images_val, images_test, labels_train, labels_val, labels_test = load_images(path)
    


    

   
    # # make patches
    builder(
        images_train, labels_train, patch_size, 'train', dataset_name, sf_range, his_range)
    builder(
        images_val, labels_val, patch_size, 'val', dataset_name, sf_range, his_range)
    builder(
        images_test, labels_test, patch_size, 'test', dataset_name, sf_range, his_range)

   
    # # # dct
    # process(train_patches, sf_range, his_range, 'train', dataset_name)
    # process(val_patches, sf_range, his_range, 'val', dataset_name)
    # process(test_patches, sf_range, his_range, 'test', dataset_name)

   

  

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
