import json
import os
import cv2
from sys import path
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
# from cnn_test import main as test
# from dct_train import main as train
from preprocessor import Preprocessor


class Input:
    def __init__(self, downscale_factor, grayscale, dct_rep, patch_size, bands, sf_lo, sf_mid, sf_hi, his_range):
        self.downscale_factor = downscale_factor
        self.colour_space = self.get_colour_space(grayscale)
        self.dct_rep = dct_rep
        self.patch_size = patch_size
        self.bands = bands
        self.sf_range = [sf_lo, sf_mid, sf_hi]
        self.his_range = his_range

        self.sf_num = self.num_of_sf(self.sf_range, bands)
        self.dset_name = self.get_dset_name(grayscale)
        self.his_size = self.get_his_range()

    def num_of_sf(self):
        if self.multibands == 3:
            return sum([sf[1] - sf[0] for sf in self.sf_range])
        else:
            return self.sf_range[self.bands][1] - self.sf_range[self.bands][0]
       
    def get_dset_name(self, grayscale):
        return f'g:{grayscale}p:{self.patch_size}_his:{self.his_range[0]},{self.his_range[1]}_sf_num:{self.sf_num}_subbands:{self.bands}'

    def get_colour_space(self, grayscale):
        return cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB

    def get_his_range(self):
        return (len(range(self.his_range[0], self.his_range[1])) + 1) * self.sf_num



# # calculate the number of spatial frequencies being used


if __name__ == "__main__":
    with open('config.json') as f:
        args = json.load(f)

    input = Input(**args)

    preprocessor = Preprocessor(input, os.getcwd())
    
    preprocessor.dct_builder()

    # epochs = args['epochs']
    # batch_size = args['batch_size']
    # architecture = args['architecture']
    # train(dataset_name, epochs, batch_size, architecture, his_range, sf_num)




