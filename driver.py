import os
from sys import path
from input import NoiseInput, HistInput, TransformedInput

path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
path.append(f'{os.getcwd()}/dl-4-tsc')

from training.cnn_test import main as test
from training.cnn_train import main as train
from utils.load_images import load_images
from utils.preprocessor import builder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-d", "--dct", help="dct domain", action='store_const', const=1)
parser.add_argument("-n", "--noise", help="noiseprint", action='store_const', const=2)
parser.add_argument("-s", "--histogram", help="dct histogram", action='store_const', const=3)
parser.add_argument("-w", "--wavelet", help="wavelet domain", action='store_const', const=4)
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')

args = parser.parse_args()

def make_name(architecture, input_shape, epochs, batch_size):
    return f'models/cnn_{architecture}_{input_shape}_{epochs}_{batch_size}'

if __name__ == "__main__":
    classes = ['facebook', 'flickr', 'google+', 'instagram', 'original', 'telegram', 'twitter',  'whatsapp']

    h_input = HistInput(hist_rep="hist_1D", patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")

    n_input = NoiseInput(patch_size=64, domain="Noise")

    d_input = TransformedInput(0, domain="DCT")

    w_input = TransformedInput(0, domain="DWT")

    epochs = 10
    batch_size = 32
    architecture = 'dct_cnn_2017'
    location = 'dct_models'

    arguments = {args.dct: d_input, args.wavelet: w_input, args.histogram: h_input, args.noise: n_input}

    dset = load_images(classes, os.getcwd())

    for argument in arguments.items():
        if argument[0]:
            name = make_name(architecture, argument[1].input_shape, epochs, batch_size)
            if args.process:
                builder(argument[1], dset)
            elif args.train:
                train(epochs, batch_size, architecture, location, argument[1], classes, name)
            elif args.test:
                test(name, argument[1], classes)




   






