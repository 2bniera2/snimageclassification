import numpy as np
import h5py
from noiseprint2 import gen_noiseprint, normalize_noiseprint


def make_patches(im, size):
    patches = []
    for i in range(0, im.shape[0]-size+1, size):
        for j in range(0, im.shape[1]-size+1, size):
            image = im[i:i+size, j:j+size]
            patches.append(image)
    return patches


def noise_extractor(input, task, examples, labels):
    # initialise X datasets 
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, input.patch_size, input.patch_size), maxshape=(None, input.patch_size, input.patch_size))
        _ = f.create_dataset('labels', shape=(0, 2), maxshape=(None, 2))
    
    
        for im_num, (path, label) in enumerate(zip(examples, labels)):
            print(f'{im_num+1}/{len(examples)}')
            noiseprint = normalize_noiseprint(gen_noiseprint(path, quality=101))
            noiseprint_patches = make_patches(noiseprint, input.patch_size)

            for noiseprint_patch in noiseprint_patches:

                dset = f['examples']
                dset.resize((dset.shape[0] + 1, input.patch_size, input.patch_size))
                dset[-1] = noiseprint_patch

                dset = f['labels']
                dset.resize((dset.shape[0] + 1, 2))
                dset[-1] = np.array([label, im_num])

                    







