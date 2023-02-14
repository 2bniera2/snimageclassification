import numpy as np
import cv2
import h5py
import extract_dcts 

def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()

def make_patches(im, size):
    patches = []
    for i in range(0, im.shape[0]-size+1, size):
        for j in range(0, im.shape[1]-size+1, size):
            # append byte form of images
            patches.append(im_to_bytes(im[i:i+size, j:j+size]))
    return patches

def builder(paths, labels, size, task, dataset_name, sf_range, his_range):
    his_size = (len(range(his_range[0], his_range[1])) + 1) * sf_range

    # initialise y datasets
    with h5py.File(f'processed/labels_{task}_{dataset_name}.h5', 'w') as f:
        dset = f.create_dataset('facebook', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('flickr', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('google+', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('instagram', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('original', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('telegram', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('twitter', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))
        dset = f.create_dataset('whatsapp', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))

    # initialise X datasets 
    with h5py.File(f'processed/DCT_{task}_{dataset_name}.h5', 'w') as f:
        dset = f.create_dataset('facebook', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('flickr', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('google+', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('instagram', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('original', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('telegram', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('twitter', shape=(0, his_size), maxshape=(None, his_size))
        dset = f.create_dataset('whatsapp', shape=(0, his_size), maxshape=(None, his_size))
        

    # generate patches from an image and extract the dcts from each patch and store in dataset
    for index, (path, label) in enumerate(zip(paths, labels)):
        with h5py.File(f'processed/DCT_{task}_{dataset_name}.h5', 'a') as DCTs , h5py.File(f'processed/labels_{task}_{dataset_name}.h5', 'a') as Labels:

            # get patches from images
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            patches = make_patches(image, size)

            # extract dct histograms from each patch 
            patch_histograms = extract_dcts(patches, sf_range, his_range, task, dataset_name)


            #iterate over all patches
            for patch_histogram in patch_histograms:

                labels_dset = Labels[label]
                dct_dset = DCTs[label]

                labels_dset.resize(labels_dset.shape[0] + 1, axis=0)
                dct_dset.resize((dset.shape[0] + 1, his_size))

                dset[-1] = patch_histogram


                dset[-1] = f"{label}.{index}" if task == 'test' else label



 