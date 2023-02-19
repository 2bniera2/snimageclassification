from sys import path
import os
path.append(f'{os.getcwd()}/training')
path.append(f'{os.getcwd()}/utils')
path.append(f'{os.getcwd()}/noiseprint2')

import json
from preprocessor import main as preprocess
from dct_train import main as train
from cnn_test import main as test

# calculate the number of spatial frequencies being used
def num_of_sf(sf_range, use_subbands):
    if use_subbands is "all":
        return sum([sf[1] - sf[0] for sf in sf_range])
    else:
        return sf_range[1] - sf_range[0]


def main():

    with open('config.json') as f:
        args = json.load(f)

    # dictates transformations applied to image before generating patches
    downscale_factor = args['downscale_factor']
    grayscale = args['grayscale']

    patch_size = args['patch_size']

    # determines what the cnn input should be
    dct_representation = args['dct_rep']
    his_range = args['his_range']
    # determines whether to use all subbands or just one
    use_subbands = args['use_subbands']

    if use_subbands is "all":
        sf_range = [args['sf_lo'], args['sf_mid'], args['sf_hi']]
    else: 
        sf_range = args[use_subbands]

    
    sf_num = num_of_sf(sf_range, use_subbands)

    dataset_name = f'g:{grayscale}p:{patch_size}_his:{his_range[0]},{his_range[1]}_sf_num:{sf_num}_subbands:{use_subbands}'

    preprocess(
        patch_size,
        dataset_name,
        his_range,
        sf_range,
        use_subbands,
        downscale_factor,
        grayscale
    )


    epochs = args['epochs']
    batch_size = args['batch_size']
    architecture = args['architecture']
    model_name = f'{architecture}_e:{epochs}_bs:{batch_size}'
    train(model_name, dataset_name, epochs, batch_size, architecture, his_range, sf_num)

    # test(model_name, dataset_name, '')


if __name__ == "__main__":
    main()


