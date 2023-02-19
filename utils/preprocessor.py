from utils.load_images import load_images
from utils.make_patches import builder
from sys import path
import os

path.append(f'{os.getcwd()}/../noiseprint2')
from utils.dataset_builder import noise_extractor



def main(patch_size, dataset_name, his_range, sf_range, use_subbands):
    path = f'{os.getcwd()}'

    # load images and split data
    images_train, images_val, images_test, labels_train, labels_val, labels_test = load_images(path)

    tasks = {
        'train': [images_train, labels_train],
        'val': [images_val, labels_val],
        'test': [images_test, labels_test]
    }
    

    for task in tasks.items():
    # make patches and extract dct coefficients
        builder(
            task[1][0],
            task[1][1],
            patch_size, 
            task[0],
            dataset_name,
            sf_range, 
            his_range
        )


        
    # # noise
    # d_name = f'p:{patch_size}'
    # noise_extractor(
    #     images_train, 'train', patch_size, d_name
    # )

    # noise_extractor(
    #     images_val, 'val', patch_size, d_name
    # )

    # noise_extractor(
    #     images_test, 'test', patch_size, d_name
    # )


    

if __name__ == "__main__":
    main()



