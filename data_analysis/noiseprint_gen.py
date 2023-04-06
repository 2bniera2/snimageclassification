from noiseprint2 import gen_noiseprint, normalize_noiseprint
import os
import numpy as np
import h5py



class_dict = {
    'facebook':[],
    'instagram':[],
    'orig':[],
    'telegram':[],
    'twitter':[],
    'whatsapp':[]
}


for CLASS in os.listdir(f'{os.getcwd()}/sample'):
    ims = []

    for IMAGE in sorted(os.listdir(f'{os.getcwd()}/sample/{CLASS}')):

        path = f'{os.getcwd()}/sample/{CLASS}/{IMAGE}'
        noiseprint = normalize_noiseprint(gen_noiseprint(path, quality=101))
        ims.append(noiseprint)
        
    np.save(f'{os.getcwd()}/noiseprint_extracted/{CLASS}', np.array(ims))

            


                

