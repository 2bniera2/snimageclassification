from input import Input
import os, argparse
from sys import path

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
path.append(f'{os.getcwd()}/cnn_single')
path.append(f'{os.getcwd()}/models')

from utils.load_fodb import load_fodb
from utils.preprocessor import builder
from cnn_single.cnn_train import train
from cnn_single.cnn_test import test

# parse cli args to pick preprocessing, train, test and domain
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')
args = parser.parse_args()

# input variables
classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'fodb'

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
    train(epochs, batch_size, architecture, location, input, classes, name)
if args.test:
    test(name, input, classes)
