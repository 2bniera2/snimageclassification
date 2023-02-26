import os
from sys import path
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
import cv2
from cnn_test import main as test
from cnn_train import main as train
from preprocessor import Preprocessor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dct", help="preprocess images to dct domain", action='store_true')
parser.add_argument("-n", "--noise", help="preprocess images to noise residuals", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')

args = parser.parse_args()



class Input:
    def __init__(self, grayscale, dct_rep, patch_size, band_mode, sf_lo, sf_mid, sf_hi, his_range, domain):
        self.domain = domain
        self.colour_space = self.get_colour_space(grayscale)
        self.dct_rep = dct_rep
        self.patch_size = patch_size
        self.band_mode = band_mode
        self.sf_range = [sf_lo, sf_mid, sf_hi]
        self.his_range = his_range
        self.sf_num = self.num_of_sf()
        self.dset_name = self.get_dset_name(grayscale)
        self.his_size = self.get_his_range()
        self.input_shape = self.get_input_shape()

    def num_of_sf(self):
        if self.band_mode == 3:
            return sum([sf[1] - sf[0] for sf in self.sf_range])
        else:
            return self.sf_range[self.band_mode][1] - self.sf_range[self.band_mode][0]
       
    def get_dset_name(self, grayscale):
        if self.domain == 'DCT':
            return f'g:{grayscale}_p:{self.patch_size}_his:{self.his_range[0]},{self.his_range[1]}_sfnum:{self.sf_num}_band_mode:{self.band_mode}'
        elif self.domain == 'Noise':
            return f'p:{self.patch_size}'

    def get_colour_space(self, grayscale):
        return cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB

    def get_his_range(self):
        return (len(range(self.his_range[0], self.his_range[1])) + 1) * self.sf_num

    def get_input_shape(self):
        if self.domain == 'DCT':
            return (self.his_size, 1)
        elif self.domain == 'Noise':
            return (self.patch_size, self.patch_size, 1)


if __name__ == "__main__":

    input = Input(
        grayscale=False,
        dct_rep="hist_1D",
        patch_size=64,
        band_mode=0,
        sf_lo=[1, 10],
        sf_mid=[11, 20],
        sf_hi=[20, 30],
        his_range=[-50, 50],
        domain='DCT'
    )
    preprocessor = Preprocessor(input, os.getcwd())

    if args.dct:
        preprocessor.dct_builder()
    if args.noise:
        preprocessor.noise_builder()

        
    epochs = 10
    batch_size = 32
    architecture = 'dct_cnn_2017'

    if args.train:
        train(epochs, batch_size, architecture, input)
    if args.test:
        test(input, epochs, batch_size, architecture)







