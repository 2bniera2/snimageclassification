import os
from sys import path
from input import TransformedInput

path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')

from training.image_cnn_test import main as test
from training.image_cnn_train import main as train
from utils.load_iplab import load_iplab
from utils.load_fodb import load_fodb

from utils.preprocessor import builder

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')

args = parser.parse_args()


def make_name(architecture, input_shape, epochs, batch_size):
    return f'models/cnn_{architecture}_{input_shape}_{epochs}_{batch_size}'



if __name__ == "__main__":
    classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']



    epochs = 10
    batch_size = 16
    architecture = 'vgg_16'

    dataset = 'fodb'

    if dataset=='fodb':
        dset = load_fodb(classes, os.getcwd(), dataset)
    elif dataset=='iplab':
        dset = load_iplab(classes, os.getcwd(), dataset)

    input_shape = (224, 224)

    d_input = TransformedInput(input_shape=input_shape, dset_name=dataset, domain="DCT")

    w_input = TransformedInput(input_shape=input_shape, dset_name=dataset, domain="DWT")

    input = d_input
    
    name = make_name(architecture, input.input_shape, epochs, batch_size)
    if args.process:
        builder(input, dset)

    if args.train:
        train(epochs, batch_size, architecture, input, classes, name)

    if args.test:
        test(name, input, classes)






