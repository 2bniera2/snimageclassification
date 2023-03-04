from utils.histogram_extractor import histogram_extractor
from utils.noise_extractor import noise_extractor
from utils.transform_builder import transform_builder

from sys import path
import os

path.append(f'{os.getcwd()}/../noiseprint2')

options = {
    'Histogram': histogram_extractor,
    'Noise':    noise_extractor,
}

def builder(input, dset, domain):
    for task, dset, in dset.items():
        options.get(domain, transform_builder)(input, task, dset[0], dset[1])

