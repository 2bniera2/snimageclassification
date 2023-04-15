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
args = parser.parse_args()

# list of classes and dataset choice
classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'fodb'

def _2D_input():
    # named tuples to store input state
    DCTInput = namedtuple("DCTInput", "input_shape dset_name domain")
    HistInput = namedtuple("HistInput", "sf his_range input_shape dset_name domain")

    # input parameters for preprocessing
    d_input = DCTInput((224, 224), " ", "DCT")
    h_input = HistInput(sf=[1,64], his_range=[-100, 100], input_shape=None, dset_name=None, domain="2DHist")
    h_input = input._replace(input_shape=((h_input.his_range[1] - h_input.his_range[0]), (h_input.sf[1] - h_input.sf[0])))
    input = h_input
    dset_name = f'{dataset}_{input.input_shape}'
    input = input._replace(dset_name=dset_name)

    # model hyperparameters
    epochs = 10
    batch_size = 32
    architecture = 'vgg16'
    trainable = True
    regularize = False
    optimizer = optimizers.Adam(learning_rate=0.0001)
    weights="imagenet"
    ModelInput = namedtuple("ModelInput", "architecture trainable regularize optimizer weights epochs batch_size")
    model_input = ModelInput(architecture, trainable, regularize, optimizer, weights, epochs, batch_size)
    name = f'{architecture}_{input.input_shape}_{epochs}_{batch_size}_{trainable}_{regularize}_{optimizer._name}_{weights}'

    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)
    if args.train:
        _2d_train(input, model_input, classes, name)
    if args.test:
        test(name, input, classes)


def _1D_input():
    # preprocessing parameters
    h_input = Input(dataset, patch_size=64, sf=[1,10], his_range=[-50, 50], domain="Histogram")
    n_input = Input(dataset, domain="Noise", patch_size=64)
    input = h_input

    # model hyperparameters
    epochs = 10
    batch_size = 32
    architecture = 'prnu_cnn'
    location = 'noise_models'
    name = f'{architecture}_{input.model_name}_{epochs}_{batch_size}'

    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)
    if args.train:
        _1d_train(epochs, batch_size, architecture, location, input, classes, name)
    if args.test:
        test(name, input, classes)

def fusion_input():
    h_input = Input(dataset, patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")
    n_input = Input(dataset, domain="Noise", patch_size=64)

    epochs = 10
    batch_size = 16
    architecture = 'FusionNET'

    name = f'{architecture}_{epochs}_{batch_size}'

    if args.train:
        fusion_train(epochs, batch_size, architecture, h_input, n_input, classes, name)
    if args.test:
        test(name, h_input, n_input, classes)


def _1D_input_alt():
    # named tuple to store input state
    PInput = namedtuple("PInput", "sf his_range domain dset_name input_shape")
    input = PInput(sf=[1,10], his_range=[-50, 50], domain="Patchless", dset_name=None, input_shape=None)
    dset_name = f'patchless_{input.sf[0]},{input.sf[1]}_{input.his_range[0]},{input.his_range[1]}'
    input_shape = ((input.his_range[1] - input.his_range[0] + 1) * (input.sf[1] - input.sf[0]))
    input = input._replace(dset_name=dset_name, input_shape=input_shape)
    #preprocess
    if args.process:
        dset = load_fodb(classes, os.getcwd())
        builder(input, dset)

    # model hyperparameters
    epochs = 10
    batch_size = 32
    architecture = 'resnet50'
    trainable = False
    regularize = False
    optimizer = optimizers.Adam(learning_rate=0.0001)
    # optimizer = optimizers.SGD(momentum=0.9)
    weights="imagenet"

    ModelInput = namedtuple("ModelInput", "architecture trainable regularize optimizer weights epochs batch_size")

    model_input = ModelInput(architecture, trainable, regularize, optimizer, weights, epochs, batch_size)
    name = f'{architecture}_0_{epochs}_{batch_size}_{trainable}_{regularize}_{optimizer._name}_{weights}'

    if args.train:
        _1d_alt_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)



def main():
    if args._1d_hist: _1D_input()
    if args._1d_hist_alt: _1D_input_alt()
    if args.fusion: fusion_input()
    if args._2d: _2D_input()

if __name__ == "__main__": main()
