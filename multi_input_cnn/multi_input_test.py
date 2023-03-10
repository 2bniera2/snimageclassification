import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages

import numpy as np
from keras import models
from sys import path
from multi_input_data_generator import multi_input_data_generator
from test_utils import get_labels, get_indices, patch_truth, image_truth, tuple_gen, viewer





# get predictions and convert numerical values to class name
def get_predictions(input1, input2, classes, model):
    gen = multi_input_data_generator(
        f'{path[0]}/processed/{input1.dset_name}_test.h5',
        f'{path[0]}/processed/{input2.dset_name}_test.h5',
        'examples',
        'examples',
        classes,
        shuffle=False
    )

    return np.argmax(model.predict(gen, use_multiprocessing=True, workers=8), axis=1)
    


def main(model_name, input1, input2, classes):
    model = models.load_model(model_name)

    # predictions represented as integer representation of classes
    predictions = get_predictions(input1, input2, classes, model)


    # labels with class and image number
    labels = get_labels(input1)

    patch_truth(labels, predictions, classes)

    image_truth(labels, predictions, classes)

   

