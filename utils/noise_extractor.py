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
    with h5py.File(f'processed/Noise_{task}_{input.dset_name}.h5', 'w') as f:
        _ = f.create_dataset('Noise', shape=(0, input.patch_size, input.patch_size), maxshape=(None, input.patch_size, input.patch_size))
        _ = f.create_dataset('labels', shape=(0, 2), maxshape=(None, 2))
    
    
        for im_num, (path, label) in enumerate(zip(examples, labels)):
            noiseprint = normalize_noiseprint(gen_noiseprint(path, quality=101))
            print(noiseprint.shape)
            noiseprint_patches = make_patches(noiseprint, input.patch_size)
            print(len(noiseprint_patches))

            for noiseprint_patch in noiseprint_patches:

                dset = f['Noise']
                dset.resize((dset.shape[0] + 1, input.patch_size, input.patch_size))
                dset[-1] = noiseprint_patch

                dset = f['labels']
                dset.resize((dset.shape[0] + 1, 2))
                dset[-1] = np.array([label, im_num])

                    







