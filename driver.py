import os
from sys import path
from input import NoiseInput, HistInput

path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')

from training.cnn_test import main as test
from training.cnn_train import main as train
from training.multi_input_train import main as multi_train
from utils.load_iplab import load_images
from utils.preprocessor import builder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", help="preprocess flag", action='store_true')
parser.add_argument("-n", "--noise", help="noiseprint", action='store_const', const=1)
parser.add_argument("-s", "--histogram", help="dct histogram", action='store_const', const=2)
parser.add_argument("-t", "--train", help="train model", action='store_true')
parser.add_argument("-e", "--test", help="evaluate model", action='store_true')


args = parser.parse_args()

def make_name(architecture, input_shape, epochs, batch_size):
    return f'models/cnn_{architecture}_{input_shape}_{epochs}_{batch_size}'

if __name__ == "__main__":
    classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']

    h_input = HistInput(hist_rep="hist_1D", patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")

    n_input = NoiseInput(patch_size=64, domain="Noise")



    epochs = 20
    batch_size = 20
    architecture = 'dct_cnn_2017'
    location = 'dct_models'

    arguments = {args.histogram: h_input, args.noise: n_input}
    
    dset = load_images(classes, os.getcwd())

    for argument in arguments.items():
        if argument[0]:
            name = make_name(architecture, argument[1].input_shape, epochs, batch_size)
            if args.process:
                builder(argument[1], dset)
            elif args.train:
                train(epochs, batch_size, architecture, location, argument[1], classes, name)
            elif args.test:
                test(name, argument[1], classes)




   






