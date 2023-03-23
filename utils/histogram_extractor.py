import numpy as np
import h5py
import utils._1d_histograms as _1d_histograms
import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# convert patch to bytes so jpeg2dct can load it
def im_to_bytes(patch, q):
    buf = io.BytesIO()
    patch.save(buf, format='JPEG', qtables=q)
    return buf.getvalue()

# from image, create a list of patches of defined size
def make_patches(image, patch_size, q=None, to_bytes=True):
    patches = []
    indices = []
    for i in range(0, image.width-patch_size+1, patch_size):
        for j in range(0, image.height-patch_size+1, patch_size):
            patch = image.crop((i, j, i+patch_size, j+patch_size))
            indices.append((i//64, j//64))
            if to_bytes:
                patch = im_to_bytes(patch, q)
            patches.append(patch)
    return patches, indices    


def histogram_extractor(input, task, examples, labels):
    # initialise dataset 
    with h5py.File(f'processed/{input.dset_name}_{task}.h5', 'w') as f:
        _ = f.create_dataset('examples', shape=(0, input.his_shape), maxshape=(None, input.his_shape))
        _ = f.create_dataset('labels', shape=(0, 2), maxshape=(None, 2))
        _ = f.create_dataset('indices', shape=(0, 2), maxshape=(None, 2))

        # generate patches from an image and extract the dcts from each patch and store in dataset
        for im_num, (path, label) in enumerate(zip(examples, labels)):
                print(f'{im_num+1}/{len(examples)}')

                # load image
                image = Image.open(path)

                # get q tables
                qtable = image.quantization

                # break down image to patches
                if input.patch_size:
                    patches, indices = make_patches(image, input.patch_size, qtable, True)

                # extract dct histograms from each patch 
                patch_histograms = _1d_histograms.process(patches, input)

                #iterate over all patches and save to dataset
                for i, patch_histogram in enumerate(patch_histograms):
                    dct_dset = f['examples']
                    dct_dset.resize((dct_dset.shape[0] + 1, input.his_shape))
                    dct_dset[-1] = patch_histogram
                    
                    labels_dset = f['labels']
                    labels_dset.resize((labels_dset.shape[0] + 1, 2))
                    labels_dset[-1] = np.array([label, im_num])

                    vis_dset = f['indices']
                    vis_dset.resize((vis_dset.shape[0] + 1, 2))
                    vis_dset[-1] = np.array([indices[i][0], indices[i][1]])
