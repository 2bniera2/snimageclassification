import os
from sys import path
from input import NoiseInput, HistInput, TransformedInput
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
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-d", "--dct", help="dct domain", action='store_true')
parser.add_argument("-n", "--noise", help="noiseprint", action='store_true')
parser.add_argument("-h", "--histogram", help="dct histogram", action='store_true')
parser.add_argument("-w", "--wavelet", help="wavelet domain", action='store_true')
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

    h_input = HistInput(
        hist_rep="hist_2D",
        patch_size=64,
        sf=[1, 10],
        his_range=[-50, 50],
        domain="DCT"
    )

    n_input = NoiseInput(
        patch_size=64,
        domain="Noise"
    )

    d_input = TransformedInput(
        domain="DCT"
    )

    w_input = TransformedInput(
        domain="DWT"
    )

    if args.process:
        dset = load_images(classes, os.getcwd())
        if args.dct: 
            builder(d_input, dset)
        if args.wavelet:
            builder(w_input, dset)
        if args.histogram:    
            builder(h_input, dset)
        if args.noise:
            builder(n_input, dset)

    epochs = 10
    batch_size = 32
    architecture = 'dct_cnn_2017'

    if args.dct:
        train_test(d_input, architecture, epochs, batch_size, classes)
    if args.noise:
        train_test(n_input, architecture, epochs, batch_size, classes)
    if args.wavelet:
        train_test(w_input, architecture, epochs, batch_size, classes)
    if args.histogram:
        train_test(h_input, architecture, epochs, batch_size, classes)



   






