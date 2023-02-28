import os
from sys import path
from input import Input
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
path.append(f'{os.getcwd()}/dl-4-tsc')

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

if __name__ == "__main__":
    classes = [
        'facebook',
        'flickr',
        'google+',
        'instagram',
        'original',
        'telegram',
        'twitter', 
        'whatsapp'
    ]

    input = Input(
        dct_rep="hist_1D",
        patch_size=64,
        band_mode=0,
        sf_lo=[1, 10],
        sf_mid=[11, 29],
        sf_hi=[30, 37],
        his_range=[-50, 50],
        domain='DCT',
        classes=classes
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







