from collections import namedtuple
from input import Input

import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
from sys import path

path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/training')

from utils.load_fodb import load_fodb
from utils.load_iplab import load_iplab
from utils.preprocessor import builder
from training._1d_hist_train import train as _1d_train
from training.noise_train import train as noise_train
from training._1d_hist_alt_train import train as _1d_alt_train
from training.fusion_train import main as fusion_train
from training.transfer_train import main as transfer_train

from training._2d_train import main as _2d_train
from evaluate_models import test

from keras import optimizers

# parse cli arguments
parser = argparse.ArgumentParser()
# CLI args for the tasks to do
parser.add_argument("--process", help="preprocess flag", action='store_true')
parser.add_argument("--train", help="train model", action='store_true')
parser.add_argument("--test", help="evaluate model", action='store_true')

# CLI args for the scheme to use
parser.add_argument("--_1d", help="1D Histogram", action='store_true')
parser.add_argument("--_1d_alt", help="1D Histogram Alternative", action='store_true')
parser.add_argument("--fusion", help="Fusion Model", action='store_true')
parser.add_argument("--_2d", help="2D Histogram", action='store_true')
parser.add_argument("--noise", help="Noise extraction", action='store_true')
parser.add_argument("--transform", help="DCT transform", action='store_true')
parser.add_argument("--noise_alt", help="Noiseprint alternative", action='store_true')


args = parser.parse_args()

# list of classes and dataset choice
classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']

dataset = 'fodb'

# model hyperparameters that can be modified for training
epochs = 10
batch_size = 32
learning_rate = 0.0001
optimizer = optimizers.Adam(learning_rate=learning_rate)
architecture = 'resnet50'

# out of 'notl' for no transfer learning, 'tl' for just transfer learning and 'tl+ft' for transfer learning and finetuning
# if using transfer learning, imageNET weights are used.
# if also using finetuning, there is initial finetuning session which freezes pretrained model and trains only dense layers
# then we train again but unfreeze everything  
setup = 'notl'

# build a tuple for storing model hyperparameters
ModelInput = namedtuple("ModelInput", "architecture optimizer epochs batch_size setup")
model_input = ModelInput(architecture, optimizer, epochs, batch_size, setup)


def _1D_input_alt(dset):
    """
        Function responsible for the processing, training or testing of the alternative 1 dimensional histogram scheme,
        a patchless scheme that uses imageNET models.
        Hyper parameters are tuned at the top with the following being the epochs, batch size, optimizer 
        and model.

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """
    # named tuple to store input state
    PInput = namedtuple("PInput", "sf his_range domain dset_name input_shape dset")
    input = PInput(sf=[1,10], his_range=[-100, 100], domain="Patchless", dset_name=None, input_shape=None, dset=dataset)
    dset_name = f'{dataset}_patchless_{input.sf[0]},{input.sf[1]}_{input.his_range[0]},{input.his_range[1]}'
    input_shape = ((input.his_range[1] - input.his_range[0] + 1) * (input.sf[1] - input.sf[0]))
    input = input._replace(dset_name=dset_name, input_shape=input_shape)
   
    if args.process:
        builder(input, dset)

    name = f'{architecture}_patchless_{epochs}_{batch_size}_{optimizer._name}_{learning_rate}_{dataset}'

    if args.train:
        _1d_alt_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)


def _2D_input(dset):
    """
        Function responsible for the processing, training or testing of the alternative 2 dimensional histogram scheme,
        a scheme loosely based off of the 1D patchless histogram alternative scheme utilising imageNET models.
        Hyper parameters are tuned at the top with the following being the epochs, batch size, optimizer 
        and model.

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """

    # named tuples to store input state
    HistInput = namedtuple("HistInput", "sf his_range input_shape dset_name domain dset")

    # input parameters for preprocessing
    input = HistInput(sf=[1,64], his_range=[-100, 100], input_shape=None, dset_name=None, domain="2DHist", dset=dataset)
    input = input._replace(input_shape=((input.sf[1] - input.sf[0]),(input.his_range[1] - input.his_range[0] + 1)))
    dset_name = f'{dataset}_{input.input_shape}'
    input = input._replace(dset_name=dset_name)

    if args.process:
        builder(input, dset)

    name = f'{architecture}_2d_{epochs}_{batch_size}_{optimizer._name}_{learning_rate}_{dataset}'
    if args.train:
        _2d_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)


def dct_transform(dset):
    """
        Function responsible for the processing, training or testing of the DCT transform scheme,
        each image is transformed into the DCT domain.
        Hyper parameters are tuned at the top with the following being the epochs, batch size, optimizer, setup
        and model.

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """

    # named tuples to store input state
    DCTInput = namedtuple("DCTInput", "input_shape dset_name domain")

    # input parameters for preprocessing
    input = DCTInput((224, 224), " ", "DCT")
    dset_name = f'{dataset}_{input.input_shape}'
    input = input._replace(dset_name=dset_name)

    if args.process:
        builder(input, dset)

    name = f'{architecture}_dct_{epochs}_{batch_size}_{optimizer._name}_{learning_rate}_{dataset}_{setup}'

    if args.train:
        transfer_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)



def noise_alt_input(dset):
    """
        Function responsible for the processing, training or testing of the alternative Noiseprint scheme, where we
        extract the noiseprint from each image and resize before storing.
        Hyper parameters are tuned at the top with the following being the epochs, batch size, optimizer, setup, 

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """

    # named tuples to store input state
    NoiseInput = namedtuple("NoiseInput", "input_shape dset_name domain")

    # input parameters for preprocessing
    input = NoiseInput((224, 224), " ", "Noise_alt")
    dset_name = f'noise_{dataset}_{input.input_shape}'
    input = input._replace(dset_name=dset_name)

    if args.process:
        builder(input, dset)

    name = f'{architecture}_noise_{epochs}_{batch_size}_{optimizer._name}_{learning_rate}_{dataset}_{setup}'

    if args.train:
        transfer_train(input, model_input, classes, name)
    if args.test:
        test(name, input, None, classes)

        
def _1D_input(dset):
    """
        Function responsible for the processing, training or testing of the original 1D histogram scheme
        Hyper parameters are tuned at the top with the following being the epochs, batch size.

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """

    # preprocessing parameters
    input = Input(dataset, patch_size=0, sf=[1,10], his_range=[-50, 50], domain="Histogram")

    if args.process:
        builder(input, dset)

    architecture = 'dct_cnn'
    location = 'dct_models'

    name = f'{architecture}_{input.model_name}_{epochs}_{batch_size}'
    if args.train:
        noise_train(epochs, batch_size, architecture, location, input, classes, name)
    if args.test:
        test(name, input, None, classes)


def noise_input(dset):
    """
        Function responsible for the processing, training or testing of the noiseprint scheme,
        each image has its noiseprint extracted and patchified.
        Hyper parameters are tuned at the top with the following being the epochs, batch size.

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """

    input = Input(dataset, domain="Noise", patch_size=64)

    if args.process:
        builder(input, dset)

    architecture = 'prnu_cnn'
    location = 'noise_models'
    name = f'{architecture}_{input.model_name}_{epochs}_{batch_size}'

    if args.train:
        _1d_train(epochs, batch_size, architecture, location, input, classes, name)
    if args.test:
        test(name, input, None, classes)



def fusion_input():
    """
        A hybrid between the 1D histogram scheme and noiseprint scheme.
        Hyper parameters are tuned at the top with the following being the epochs, batch size.

    Args:
        dset (_type_): A dictionary of subsets of the dataset (training, test and split)
    """

    h_input = Input(dataset, patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")
    n_input = Input(dataset, domain="Noise", patch_size=64)

    architecture = 'FusionNET'

    name = f'{architecture}_{epochs}_{batch_size}'

    if args.train:
        fusion_train(epochs, batch_size, architecture, h_input, n_input, classes, name)
    if args.test:
        test(name, h_input, n_input, classes)



def main():
    if dataset == 'iplab':
        dset = load_iplab(classes, os.getcwd())
    if dataset == 'fodb':
        dset = load_fodb(classes, os.getcwd())

      

    if args._1d: _1D_input(dset)
    if args.noise: noise_input(dset)
    if args.fusion: fusion_input()
    if args._1d_alt: _1D_input_alt(dset)
    if args._2d: _2D_input(dset)
    if args.transform: dct_transform(dset)
    if args.noise_alt: noise_alt_input(dset)

if __name__ == "__main__": main()
