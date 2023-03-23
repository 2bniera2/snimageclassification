from collections import namedtuple
import os, argparse
from sys import path

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/cnn_transfer_learning')

from cnn_transfer_learning.cnn_test import main as test
from cnn_transfer_learning.cnn_train import main as train
from utils.load_iplab import load_iplab
from utils.load_fodb import load_fodb
from utils.preprocessor import builder

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')
args = parser.parse_args()

Input = namedtuple("Input", "input_shape dset_name domain")

classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'fodb'
epochs = 10
batch_size = 32
architecture = 'resnet50'
d_input = Input((224, 224), " ", "DCT")
h_input = Input((201, 62), " ", "2DHist")
input = h_input

dset_name = f'{dataset}_{input.input_shape}'

input._replace(dset_name=dset_name)

name = f'cnn_{architecture}_{input.input_shape}_{epochs}_{batch_size}'
path = f'model/{name}/{name}'

if args.process:
    dset = {'fodb': load_fodb, 'iplab': load_iplab}.get(dataset, load_fodb)(classes, os.getcwd())
    builder(input, dset)

if args.train:
    train(epochs, batch_size, architecture, input, classes, path)
if args.test:
    test(path, input, classes)