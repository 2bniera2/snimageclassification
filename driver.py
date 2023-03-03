import os
from sys import path
from input import DCTInput, NoiseInput
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
path.append(f'{os.getcwd()}/dl-4-tsc')

import cv2
from cnn_test import main as test
from cnn_train import main as train
from utils.load_images import load_images
from utils.preprocessor import builder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess data to desired domain", action='store_true')
parser.add_argument("-d", "--dct", help="dct train or test", action='store_true')
parser.add_argument("-n", "--noise", help="noise train or test", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')

args = parser.parse_args()

def make_name(architecture, input_shape, epochs, batch_size):
    return f'models/cnn_{architecture}_{input_shape}_{epochs}_{batch_size}'

def train_test(input, architecture, epochs, batch_size, classes):
    name =  make_name(architecture, input.input_shape, epochs, batch_size)
    if args.train:
        train(epochs, batch_size, architecture, input, classes, name)
    if args.test:
        test(name, input, classes)

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

    d_input = DCTInput(
        dct_rep="hist_1D",
        patch_size=64,
        band_mode=0,
        sf_lo=[1, 10],
        sf_mid=[11, 29],
        sf_hi=[30, 37],
        his_range=[-50, 50],
        domain="DCT"
    )

    n_input = NoiseInput(
        patch_size=64,
        domain="Noise"
    )



    if args.process:
        dset = load_images(classes, os.getcwd())
        builder(d_input, dset, d_input.domain)
        builder(n_input, dset, n_input.domain)
        builder(d_input, dset)

    epochs = 10
    batch_size = 32
    architecture = 'dct_cnn_2017'

    if args.dct:
        train_test(d_input, architecture, epochs, batch_size, classes)
    if args.noise:
        train_test(n_input, architecture, epochs, batch_size, classes)


   






