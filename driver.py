import json
from preprocessor import main as preprocess
from training.cnn_train import main as train
from training.cnn_test import main as test
# import training.cnn_train as cnn_train
# import training.cnn_test  as cnn_test
from sys import argv


def main():

    with open(argv[1]) as f:
        args = json.load(f)

    name = args['name']
    patch_size = args['patch_size']
    his_range = args['his_range']
    sf_range = args['sf_range']
    epochs = args['epochs']
    batch_size = args['batch_size']

    # preprocess(patch_size, name, his_range, sf_range)
    # train(name, epochs, batch_size)
    test(name, '')


if __name__ == "__main__":
    main()