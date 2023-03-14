from input import Input
import os, argparse
from sys import path
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')
path.append(f'{os.getcwd()}/single_input')
from utils.load_iplab import load_iplab
from utils.load_fodb import load_fodb
from utils.preprocessor import builder
from single_input.cnn_train import train
from single_input.cnn_test import test

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-n", "--noise", help="noiseprint", action='store_const', const=1)
parser.add_argument("-d", "--histogram", help="dct histogram", action='store_const', const=2)
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')

args = parser.parse_args()

classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'iplab'
epochs = 10
batch_size = 32
architecture = 'dct_cnn_hi_dropout'
location = 'dct_models'
h_input = Input(dataset, patch_size=64, sf=[1,10], his_range=[-50, 50], domain="Histogram")
n_input = Input(dataset, domain="Noise", patch_size=64)

arguments = {args.histogram: h_input, args.noise: n_input}

for argument in arguments.items():
    if argument[0]:
        name = f'models/cnn_{architecture}_{argument[1].input_shape}_{epochs}_{batch_size}'
        if args.process:
            dset = {'fodb': load_fodb, 'iplab': load_iplab}.get(dataset, load_iplab)(classes, os.getcwd())
            builder(argument[1], dset)
        elif args.train:
            train(epochs, batch_size, architecture, location, argument[1], classes, name)
        elif args.test:
            test(name, argument[1], classes)
