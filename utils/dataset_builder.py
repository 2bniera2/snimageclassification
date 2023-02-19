import numpy as np
import h5py
from noiseprint2 import gen_noiseprint, normalize_noiseprint
from utils.make_patches import make_patches
from PIL import Image



def noise_extractor(paths, task, patch_size, dataset_name):

    # initialise X datasets 
    with h5py.File(f'processed/Noise_{task}_{dataset_name}.h5', 'w') as f:
        dset = f.create_dataset('Noise', shape=(0, patch_size, patch_size), maxshape=(None, patch_size, patch_size))

    
    
    for i, path in enumerate(paths):
        print(f'{i+1}/{len(paths)}')
        noiseprint = normalize_noiseprint(gen_noiseprint(path, quality=101))
        noiseprint_patches = make_patches(noiseprint, patch_size, False)

        with h5py.File(f'processed/Noise_{task}_{dataset_name}.h5', 'a') as Noises:
            for noiseprint_patch in noiseprint_patches:

                dset = Noises['Noise']
                dset.resize((dset.shape[0] + 1, patch_size, patch_size))
                dset[-1] = noiseprint_patch










