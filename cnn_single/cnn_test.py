import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages

import numpy as np
from keras import models
from sys import path
from utils.data_generator import data_generator
from utils.test_utils import get_labels, image_truth


# get predictions and convert numerical values to class name
def get_predictions(input, classes, model):
    gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_test.h5',
        'examples',
        classes,
        shuffle=False
    )

    pred = model.predict(gen, use_multiprocessing=True, workers=8)

    return np.argmax(pred, axis=1), pred

def test(name, input, classes):
    model = models.load_model(name)

    # predictions represented as integer representation of classes
    best, probs = get_predictions(input, classes, model)


    # labels with class and image number
    labels = get_labels(input)

    

    image_truth(labels, best, classes, name)


    