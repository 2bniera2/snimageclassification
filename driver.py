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

    
    dataset_name = args['dataset_name']
    patch_size = args['patch_size']
    his_range = args['his_range']
    sf_range = args['sf_range']
    epochs = args['epochs']
    batch_size = args['batch_size']
    architecture = args['architecture']

    dataset_name = f'p:{patch_size}_his:{his_range[0]},{his_range[1]}_sf_range:{sf_range}'
    model_name = f'{architecture}_e:{epochs}_bs:{batch_size}'

    preprocess(patch_size, dataset_name, his_range, sf_range)
    train(model_name, dataset_name, epochs, batch_size, architecture, his_range, sf_range)
    test(model_name, dataset_name, '')


if __name__ == "__main__":
    main()


