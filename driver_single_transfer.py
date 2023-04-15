from input import Input
import os, argparse
from sys import path
from collections import namedtuple

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
path.append(f'{os.getcwd()}/cnn_single_transfer')
path.append(f'{os.getcwd()}/models')

from utils.load_fodb import load_fodb
from utils.preprocessor import builder
from cnn_single_transfer.cnn_train import train

from keras import optimizers


# parse cli args to pick preprocessing, train, test and domain
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')
args = parser.parse_args()

# input variables
classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']

# named tuple to store input state
Input = namedtuple("Input", "sf his_range domain dset_name input_shape")
input = Input(sf=[1,10], his_range=[-50, 50], domain="Patchless", dset_name=None, input_shape=None)
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
architecture = 'vgg16'
trainable = True
regularize = True
optimizer = optimizers.Adam(learning_rate=0.0001)
weights="imagenet"

ModelInput = namedtuple("ModelInput", "architecture trainable regularize optimizer weights epochs batch_size")

model_input = ModelInput(architecture, trainable, regularize, optimizer, weights, epochs, batch_size)
name = f'{architecture}_0_{epochs}_{batch_size}_{trainable}_{regularize}_{optimizer._name}_{weights}'


if args.train:
    train(input, model_input, classes, name)
if args.test:
    test(name, input, classes)
