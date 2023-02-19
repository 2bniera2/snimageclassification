import numpy as np
import cv2
import h5py
import extract_dcts 

def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()

def make_patches(im, size, to_bytes=True):
    patches = []
    for i in range(0, im.shape[0]-size+1, size):
        for j in range(0, im.shape[1]-size+1, size):
            # append byte form of images
            image = im[i:i+size, j:j+size]
            if to_bytes:
                image = im_to_bytes(image)
            patches.append(image)
    return patches

def builder(paths, labels, patch_size, task, dataset_name, sf_range, his_range):
    his_size = (len(range(his_range[0], his_range[1])) + 1) * (sf_range[1] - sf_range[0])

    # initialise X datasets 
    with h5py.File(f'processed/DCT_{task}_{dataset_name}.h5', 'w') as f:
        dset = f.create_dataset('DCT', shape=(0, his_size), maxshape=(None, his_size))
    
    # initialise y datasets
    with h5py.File(f'processed/labels_{task}_{dataset_name}.h5', 'w') as f:
        dset = f.create_dataset('labels', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
   
      
        

    # generate patches from an image and extract the dcts from each patch and store in dataset
    for index, (path, label) in enumerate(zip(paths, labels)):
        with h5py.File(f'processed/DCT_{task}_{dataset_name}.h5', 'a') as DCTs , h5py.File(f'processed/labels_{task}_{dataset_name}.h5', 'a') as Labels:

            # get patches from images
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if patch_size:
                patches = make_patches(image, patch_size)

            # extract dct histograms from each patch 
            patch_histograms = extract_dcts.process(patches, sf_range, his_range)


            #iterate over all patches
            for patch_histogram in patch_histograms:

                dct_dset = DCTs['DCT']
                dct_dset.resize((dct_dset.shape[0] + 1, his_size))
                dct_dset[-1] = patch_histogram

                labels_dset = Labels['labels']
                labels_dset.resize(labels_dset.shape[0] + 1, axis=0)
                labels_dset[-1] = f"{label}.{index}" if task == 'test' else label



 