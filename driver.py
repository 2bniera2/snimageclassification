from collections import namedtuple
from input import Input

import os, argparse
from sys import path

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/training')

from utils.load_fodb import load_fodb

from utils.preprocessor import builder

from training._1d_hist_train import train as _1d_train
from training._1d_hist_alt_train import train as _1d_alt_train
from training.fusion_train import main as fusion_train
from training._2d_train import main as _2d_train
from evaluate_models import test

from keras import optimizers

# parse cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')

parser.add_argument("-a", "--_1d_hist", help="1D Histogram", action='store_true')
parser.add_argument("-b", "--_1d_hist_alt", help="1D Histogram Patchless", action='store_true')
parser.add_argument("-c", "--fusion", help="Fusion Model", action='store_true')
parser.add_argument("-d", "--_2d", help="2D Histogram", action='store_true')
parser.add_argument("-n", "--noise", help="Noise extraction", action='store_true')
parser.add_argument("-f", "--transform", help="DCT transform", action='store_true')
args = parser.parse_args()

# list of classes and dataset choice
classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'fodb'

# model hyperparameters
epochs = 10
batch_size = 16
trainable = False
regularize = True
optimizer = optimizers.Adam(learning_rate=0.0001)
weights = None
architecture = 'vgg16'

ModelInput = namedtuple("ModelInput", "architecture trainable regularize optimizer weights epochs batch_size")
model_input = ModelInput(architecture, trainable, regularize, optimizer, weights, epochs, batch_size)

def _1D_input():
    # preprocessing parameters
    input = Input(dataset, patch_size=64, sf=[1,10], his_range=[-50, 50], domain="Histogram")

    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)


    architecture = 'dct_cnn'
    location = 'dct_models'

    name = f'{architecture}_{input.model_name}_{epochs}_{batch_size}'
    if args.train:
        _1d_train(epochs, batch_size, architecture, location, input, classes, name)
    if args.test:
        test(name, input, None, classes)

def noise_input():
    input = Input(dataset, domain="Noise", patch_size=64)

    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)

    # model hyperparameters
    epochs = 10
    batch_size = 32
    architecture = 'prnu_cnn'
    location = 'noise_models'
    name = f'{architecture}_{input.model_name}_{epochs}_{batch_size}'

    if args.train:
        _1d_train(epochs, batch_size, architecture, location, input, classes, name)
    if args.test:
        test(name, input, None, classes)


def fusion_input():
    h_input = Input(dataset, patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")
    n_input = Input(dataset, domain="Noise", patch_size=64)

    architecture = 'FusionNET'

    name = f'{architecture}_{epochs}_{batch_size}'

    if args.train:
        fusion_train(epochs, batch_size, architecture, h_input, n_input, classes, name)
    if args.test:
        test(name, h_input, n_input, classes)


def _1D_input_alt():
    # named tuple to store input state
    PInput = namedtuple("PInput", "sf his_range domain dset_name input_shape dset")
    input = PInput(sf=[1,10], his_range=[-50, 50], domain="Patchless", dset_name=None, input_shape=None, dset=dataset)
    dset_name = f'{dataset}_patchless_{input.sf[0]},{input.sf[1]}_{input.his_range[0]},{input.his_range[1]}'
    input_shape = ((input.his_range[1] - input.his_range[0] + 1) * (input.sf[1] - input.sf[0]))
    input = input._replace(dset_name=dset_name, input_shape=input_shape)
   
    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)

    name = f'{architecture}_patchless_{epochs}_{batch_size}_{trainable}_{regularize}_{optimizer._name}_{weights}'

    if args.train:
        _1d_alt_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)


def _2D_input():
    # named tuples to store input state
    HistInput = namedtuple("HistInput", "sf his_range input_shape dset_name domain dset")

    # input parameters for preprocessing
    input = HistInput(sf=[1,64], his_range=[-100, 100], input_shape=None, dset_name=None, domain="2DHist", dset=dataset)
    input = input._replace(input_shape=((input.his_range[1] - input.his_range[0] + 1), (input.sf[1] - input.sf[0])))
    dset_name = f'{dataset}_{input.input_shape}'
    input = input._replace(dset_name=dset_name)

    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)

  
    name = f'{architecture}_{input.input_shape}_{epochs}_{batch_size}_{trainable}_{regularize}_{optimizer._name}_{weights}'

    if args.train:
        _2d_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)


def dct_transform():
    # named tuples to store input state
    DCTInput = namedtuple("DCTInput", "input_shape dset_name domain")

    # input parameters for preprocessing
    input = DCTInput((224, 224), " ", "DCT")
    dset_name = f'{dataset}_{input.input_shape}'
    input = input._replace(dset_name=dset_name)

    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)

    name = f'{architecture}_{input.input_shape}_{epochs}_{batch_size}_{trainable}_{regularize}_{optimizer._name}_{weights}'

    if args.train:
        _2d_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)


def main():
    if args._1d_hist: _1D_input()
    if args.noise: noise_input()
    if args.fusion: fusion_input()
    if args._1d_hist_alt: _1D_input_alt()
    if args._2d: _2D_input()
    if args.transform: dct_transform()

if __name__ == "__main__": main()
