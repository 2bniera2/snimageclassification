import cv2
import numpy as np
import pywt
import h5py


def to_dct_domain(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dct = np.uint8(cv2.dct(np.float32(im)/255)*255)
    dct = cv2.resize(dct, (224, 224))
    stacked = np.stack((dct, dct, dct))
    return stacked

def to_dwt_domain(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dwt = pywt.wavedec2(im, 'bior4.4', level=2)
    arr, slices = pywt.coeffs_to_array(dwt)
    arr = cv2.resize(arr, (224, 224))
    stacked = np.stack((arr, arr, arr))
    return stacked



def transform_builder(input, task, examples, labels):
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, *input.input_shape), maxshape=(None, *input.input_shape))
        _ = f.create_dataset('labels', shape=(0, 1), maxshape=(None, 1))


        for im_num, (path, label) in enumerate(zip(examples, labels)):
            print(f'{im_num+1}/{len(examples)}')

            if input.domain == "DCT":
                im = to_dct_domain(path)
            elif input.domain == "DWT":
                im = to_dwt_domain(path)


            dct_dset = f['examples']
            dct_dset.resize((dct_dset.shape[0] + 1, *input.input_shape))
            dct_dset[-1] = im
                    
            labels_dset = f['labels']
            labels_dset.resize((labels_dset.shape[0] + 1, 1))
            labels_dset[-1] = np.array([label])

    