from input import Input
import os, argparse
from sys import path
path.append(f'{os.getcwd()}/training')
from multi_input_cnn.cnn_train import main as train
from multi_input_cnn.cnn_test import main as test

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')
args = parser.parse_args()

classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'iplab'
h_input = Input(dataset, patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")
n_input = Input(dataset, domain="Noise", patch_size=64)
epochs = 10
batch_size = 32
architecture = 'FusionNET'
name = "models/FusionNET"

if args.train:
    train(epochs, batch_size, architecture, h_input, n_input, classes, name)
if args.test:
    test(name, h_input, n_input, classes)



