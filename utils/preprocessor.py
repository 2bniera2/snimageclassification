from utils.load_images import load_images
from utils.make_patches import builder
from sys import path
import os

path.append(f'{os.getcwd()}/../noiseprint2')
from utils.dataset_builder import noise_extractor


class Preprocessor:
    def __init__(self, input, path):
        self.input = input
        self.dset = load_images(path)

    def dct_builder(self):
        for task, dset in self.dset.items():
            builder(self.input, task, dset[0], dset[1])

    def get_noise_dset_name(self):
        return f'p:{self.input.patch_size}'

    def noise_builder(self):
        for task, dset in self.dset.items():
            noise_extractor(dset[0], task, self.input.patch_size, self.get_noise_dset_name())        

