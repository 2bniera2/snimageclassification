import cv2
import numpy as np
import pywt
import h5py

def normalise(arr):
    return (arr-np. min(arr))/(np. max(arr)-np.min(arr))

def to_dct_domain(path, input_shape):
    image = cv2.imread(path, 0)
    image = cv2.dct(np.float32(image))
    image = cv2.resize(image, input_shape)
    dct = np.stack((image, image, image)).reshape((*input_shape, 3))
    return dct

def to_dwt_domain(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dwt = pywt.wavedec2(im, 'bior4.4', level=2)
    dwt, _ = pywt.coeffs_to_array(dwt)
    dwt = cv2.resize(dwt, (224, 224))
    dwt = np.repeat(dwt[:,:,np.newaxis], 3, axis=-1)

    return dwt



def transform_builder(input, task, examples, labels):
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, *input.input_shape, 3), maxshape=(None, *input.input_shape, 3))
        _ = f.create_dataset('labels', shape=(0, 1), maxshape=(None, 1))


        for im_num, (path, label) in enumerate(zip(examples, labels)):
            print(f'{im_num+1}/{len(examples)}')
            
            if input.domain == "DCT":
                im = to_dct_domain(path, input.input_shape)
            elif input.domain == "DWT":
                im = to_dwt_domain(path)


            dct_dset = f['examples']
            dct_dset.resize((dct_dset.shape[0] + 1, *input.input_shape, 3))
            dct_dset[-1] = im
                    
            labels_dset = f['labels']
            labels_dset.resize((labels_dset.shape[0] + 1, 1))
            labels_dset[-1] = np.array([label])