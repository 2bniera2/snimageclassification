import numpy as np
import h5py
import utils._1d_histograms as _1d_histograms
from PIL import Image, ImageFile
from utils.make_patches import make_patches
ImageFile.LOAD_TRUNCATED_IMAGES = True


def histogram_extractor(input, task, examples, labels):
    # initialise hdf5 file with example, labels and indices dataset where indices is just for visualising after testing.
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, input.his_shape), maxshape=(None, input.his_shape))
        _ = f.create_dataset('labels', shape=(0, 2), maxshape=(None, 2))

        # generate patches from an image and extract the dcts from each patch and store in dataset
        for im_num, (path, label) in enumerate(zip(examples, labels)):
                print(f'{im_num+1}/{len(examples)}')

                # break down image to patches
                if input.patch_size:
                    # load image
                    image = Image.open(path)

                    # get q tables
                    qtable = image.quantization

                    patches = make_patches(image, input.patch_size, qtable, True)

                    # extract dct histograms from each patch 
                    patch_histograms = _1d_histograms.process_patches(patches, input)

                    #iterate over all patches and save to dataset
                    for patch_histogram in patch_histograms:
                        dct_dset = f['examples']
                        dct_dset.resize((dct_dset.shape[0] + 1, input.his_shape))
                        dct_dset[-1] = patch_histogram
                        
                        labels_dset = f['labels']
                        labels_dset.resize((labels_dset.shape[0] + 1, 2))
                        labels_dset[-1] = np.array([label, im_num])

                     

                # if patch_size = 0, this means we don't break down images into patches, instead we take the histogram from the entire image.
                else:   
                    histogram = _1d_histograms.process(path, input)
                    dct_dset = f['examples']
                    dct_dset.resize((dct_dset.shape[0] + 1, input.his_shape))
                    dct_dset[-1] = histogram
                    
                    labels_dset = f['labels']
                    labels_dset.resize((labels_dset.shape[0] + 1, 2))
                    labels_dset[-1] = np.array([label, im_num])

