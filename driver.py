from sys import argv, path
import os
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')

import json
from preprocessor import main as preprocess
from cnn_train import main as train
from cnn_test import main as test


def main():

    with open('config.json') as f:
        args = json.load(f)

    name = args['name']
    dataset_name = args['dataset_name']
    patch_size = args['patch_size']
    his_range = args['his_range']
    sf_range = args['sf_range']
    epochs = args['epochs']
    batch_size = args['batch_size']
    architecture = args['architecture']

    preprocess(patch_size, dataset_name, his_range, sf_range)
    train(name, dataset_name, epochs, batch_size, architecture, his_range, sf_range)
    test(name, dataset_name, '')


if __name__ == "__main__":
    main()


