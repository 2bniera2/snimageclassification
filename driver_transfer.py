from collections import namedtuple
import os, argparse
from sys import path

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/cnn_transfer_learning')

from evaluate_models import test
from cnn_transfer_learning.cnn_train import main as train
from utils.load_iplab import load_iplab
from utils.load_fodb import load_fodb
from utils.preprocessor import builder

from keras import optimizers


# parse cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')
args = parser.parse_args()

# list of classes and dataset choice
classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'fodb'

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
    dset = {'fodb': load_fodb, 'iplab': load_iplab}.get(dataset, load_fodb)(classes, os.getcwd())
    builder(input, dset)

if args.train:
    train(input, model_input, classes, name)
if args.test:
    test(name, input, classes)