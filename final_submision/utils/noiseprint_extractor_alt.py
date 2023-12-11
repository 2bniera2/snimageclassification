import numpy as np
import h5py
from noiseprint2 import gen_noiseprint, normalize_noiseprint
import cv2




def noise_extractor(input, task, examples, labels):
    # initialise X datasets 
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3))
        _ = f.create_dataset('labels', shape=(0, 2), maxshape=(None, 2))
    
    
        for im_num, (path, label) in enumerate(zip(examples, labels)):
            print(f'{im_num+1}/{len(examples)}')
            noiseprint = normalize_noiseprint(gen_noiseprint(path, quality=101))
            
            image = cv2.resize(noiseprint, (224, 224))

            il = [image for _ in range(3)]

   
            noise = np.stack(il)   


            noise = noise.reshape((224,224, 3))



            
            dset = f['examples']
            dset.resize((dset.shape[0] + 1, 224, 224, 3))
            dset[-1] = noise

            dset = f['labels']
            dset.resize((dset.shape[0] + 1, 2))
            dset[-1] = np.array([label, im_num])

                    







