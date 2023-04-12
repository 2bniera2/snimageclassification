from input import Input
import os, argparse
from sys import path

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/cnn_fusion_net')
path.append(f'{os.getcwd()}/models')

from cnn_fusion_net.cnn_train import main as train
from cnn_fusion_net.cnn_test import main as test

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')
args = parser.parse_args()

classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
dataset = 'fodb'
h_input = Input(dataset, patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")
n_input = Input(dataset, domain="Noise", patch_size=64)

epochs = 10
batch_size = 16
architecture = 'FusionNET'

name = f'{architecture}_{epochs}_{batch_size}'

if args.train:
    train(epochs, batch_size, architecture, h_input, n_input, classes, name)
if args.test:
    test(name, h_input, n_input, classes)



