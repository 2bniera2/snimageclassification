import numpy as np
import cv2
import h5py


def im_to_bytes(patch):
    # NEW CV2 CODE
    _, im_buf_arr = cv2.imencode(".jpeg", patch, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return im_buf_arr.tobytes()

def make_patches(im, size):
    patches = []
    for i in range(0, im.shape[0]-size+1, size):
        for j in range(0, im.shape[1]-size+1, size):
            patches.append(im[i:i+size, j:j+size])
    return patches

def builder(paths, labels, size, task, name):

    with h5py.File(f'processed/labels_{task}_{name}.h5', 'w') as f:
        dset = f.create_dataset('labels', (0, ), maxshape=(None, ), dtype=h5py.special_dtype(vlen=str))

    X_b = []


    for index, (path, label) in enumerate(zip(paths, labels)):
        with h5py.File(f'processed/labels_{task}_{name}.h5', 'a') as f:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            patches = make_patches(image, size)

            #iterate over all patches
            for patch in patches:
                X_b.append(im_to_bytes(patch))
                
                dset = f['labels']

                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = f"{label}.{index}" if task == 'test' else label
    
    return X_b