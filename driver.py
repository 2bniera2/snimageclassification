import os
import cv2
from sys import path
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
from test import test
from train import train
from preprocessor import Preprocessor


class Input:
    def __init__(self, grayscale, dct_rep, patch_size, band_mode, sf_lo, sf_mid, sf_hi, his_range, domain):
        self.colour_space = self.get_colour_space(grayscale)
        self.dct_rep = dct_rep
        self.patch_size = patch_size
        self.band_mode = band_mode
        self.sf_range = [sf_lo, sf_mid, sf_hi]
        self.his_range = his_range
        self.sf_num = self.num_of_sf()
        self.dset_name = self.get_dset_name(grayscale)
        self.noise_dset_name = self.get_noise_dset_name()
        self.his_size = self.get_his_range()
        self.domain = domain
        self.input_shape = self.get_input_shape()

    def num_of_sf(self):
        if self.band_mode == 3:
            return sum([sf[1] - sf[0] for sf in self.sf_range])
        else:
            return self.sf_range[self.band_mode][1] - self.sf_range[self.band_mode][0]
       
    def get_dset_name(self, grayscale):
        return f'g:{grayscale}_p:{self.patch_size}_his:{self.his_range[0]},{self.his_range[1]}_sfnum:{self.sf_num}_band_mode:{self.band_mode}'

    def get_noise_dset_name(self):
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
        his_range=[-50, 50]
        domain='Noise'
    )

    # preprocessor = Preprocessor(input, os.getcwd())
    
    # # preprocessor.dct_builder()

    # preprocessor.noise_builder()


    epochs = 20
    batch_size = 32
    architecture = 'noise_cnn'
    name = f'{architecture}_e:{epochs}_b:{batch_size}'
    domain = 'Noise'

    train(name, epochs, batch_size, architecture, input)

    test(name, input.dset_name)




