from collections import namedtuple
import os, argparse
from sys import path
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/image_cnn')

from image_cnn.cnn_test import main as test
from image_cnn.cnn_train import main as train
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
batch_size = 20
architecture = 'vgg_16'
input_shape = (224, 224)
dset_name = f'{dataset}_{input_shape}'
d_input = Input(input_shape, dset_name, "DCT")
w_input = Input(input_shape, dset_name, "DWT")
input = d_input
name = f'models/cnn_{architecture}_{input.input_shape}_{epochs}_{batch_size}'

if args.process:
    dset = {'fodb': load_fodb, 'iplab': load_iplab}.get(dataset, load_iplab)(classes, os.getcwd())
    builder(input, dset)

if args.train:
    train(epochs, batch_size, architecture, input, classes, name)
if args.test:
    test(name, input, classes)