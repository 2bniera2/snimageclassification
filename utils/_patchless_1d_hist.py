from jpeg2dct.numpy import load
import numpy as np
import h5py
from utils.histogram_extract import histogram_extract




def hist(path, input):
    dct, _, _ = load(path)

    # this is just to stop numba complaining 
    his_range = (input.his_range[0], input.his_range[1])
    sf = (input.sf[0], input.sf[1])

    his =  histogram_extract(dct, sf, his_range).flatten()
    his = np.stack((his, his)).reshape((input.input_shape, 2))

    return his



def hist_builder_p(input, task, examples, labels):
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, input.input_shape, 2), maxshape=(None, input.input_shape, 2))
        _ = f.create_dataset('labels', shape=(0, 1), maxshape=(None, 1))

        for im_num, (path, label) in enumerate(zip(examples, labels)):
            print(f'{im_num+1}/{len(examples)}')
            
            
            im = hist(path, input)

            dct_dset = f['examples']
            dct_dset.resize((dct_dset.shape[0] + 1, input.input_shape, 2))
            dct_dset[-1] = im
                    
            labels_dset = f['labels']
            labels_dset.resize((labels_dset.shape[0] + 1, 1))
            labels_dset[-1] = np.array([label])