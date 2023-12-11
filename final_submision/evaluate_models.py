import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages
import numpy as np
from keras import models
from utils.data_generator import data_generator
from utils.multi_input_data_generator import multi_input_data_generator
from utils.test_utils import truth, image_truth
import h5py

def get_labels(input):
    """
        For a dataset, get the ground truth labels y
    Args:
        input (_type_): dataset parameters

    Returns:
        _type_: array of ground truth labels y
    """

    file = f'processed/{input.dset_name}_test.h5'
    with h5py.File(file, 'r') as f:
        return np.array(f['labels'][()])

# get predictions and convert numerical values to class name
def get_predictions(input, classes, model):
    """
        For a model, pass in the test set, and obtain 
        the predictions y_hat
    Args:
        input (_type_): dataset parameters
        classes (_type_): dataset classes
        model (_type_): trained model

    Returns:
        _type_: array of predictions y_hat
    """
    file =  f'processed/{input.dset_name}_test.h5'
    gen = data_generator(
        file,
        'examples',
        classes,
        shuffle=False
    )

    pred = model.predict(gen, use_multiprocessing=True, workers=8)
    return np.argmax(pred, axis=1)

def get_fusion_predictions(input1, input2, classes, model):  
    """
        For a fusion model, passs in 2 test sets, and obtain 
        the predictions y_hat
    Args:
        input1 (_type_): dataset parameters for first input
        input2 (_type_): dataset parameters for second input
        classes (_type_): dataset classes
        model (_type_): trained model

    Returns:
        _type_: array of predictions y_hat
    """

    file1 = f'processed/{input1.dset_name}_test.h5' 
    file2 = f'processed/{input2.dset_name}_test.h5' 

    gen = multi_input_data_generator(
        file1,
        file2,
        'examples',
        'examples',
        classes,
        shuffle=False
    )

    return np.argmax(model.predict(gen, use_multiprocessing=True, workers=8), axis=1)

def test(name, input1, input2, classes):
    """
        Evaluation method responsible for obtaining model accuracies from test set
    Args:
        name (_type_): name of the model
        input1 (_type_): dataset parameters for first input
        input2 (_type_): dataset parameters for second input (if using fusion model)
        classes (_type_): dataset classes
    """

    model_name = f'models/{name}'
    model = models.load_model(model_name)

    if not input2:
        predictions = get_predictions(input1, classes, model)
    elif input2:
        predictions = get_fusion_predictions(input1, input2, classes, model)
    
    # labels with class and image number
    labels = get_labels(input1)

    # obtain the accuracy and confusion matrix
    if input1.domain in ('DCT', '2DHist', 'Patchless', 'Noise_alt'):
        truth(labels, predictions, classes, f'{name}_{input1.dset_name}')
    else: 
        image_truth(labels, predictions, classes, f'{name}_{input1.dset_name}')

   
