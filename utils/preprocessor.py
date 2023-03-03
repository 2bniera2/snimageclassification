from utils.make_patches import builder
from utils.noise_extractor import noise_extractor
from utils.to_dct_domain import dset_builder

from sys import path
import os

path.append(f'{os.getcwd()}/../noiseprint2')

options = {
    'DCT': builder,
    'Noise':    noise_extractor,
    'DCT2D' : dset_builder
}

def builder(input, dset, domain="DCT2D"):
    for task, dset, in dset.items():
        options[domain](input, task, dset[0], dset[1])

