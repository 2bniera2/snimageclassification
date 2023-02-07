from sys import argv, path
import os
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')

import json
from preprocessor import main as preprocess
from cnn_train import main as train
from cnn_test import main as test


def main():

    with open(argv[1]) as f:
        args = json.load(f)

    name = args['name']
    patch_size = args['patch_size']
    his_range = args['his_range']
    sf_range = args['sf_range']
    epochs = args['epochs']
    batch_size = args['batch_size']
    architecture = args['architecture']

    # preprocess(patch_size, name, his_range, sf_range)
    train(name, epochs, batch_size, architecture)
    test(name, '')


if __name__ == "__main__":
    main()