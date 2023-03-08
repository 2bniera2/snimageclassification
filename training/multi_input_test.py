import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sys import path
import h5py
from data_generator import data_generator



def get_labels(input):
    with h5py.File(f'processed/{input.dset_name}_test.h5', 'r') as f:
        return np.array(f['labels'][()])


# get predictions and convert numerical values to class name
def get_predictions(input, classes, model):
    gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_test.h5',
        'examples',
        classes,
        shuffle=False
    )

    pred = model.predict(gen, use_multiprocessing=True, workers=8)
    return np.argmax(pred, axis=1)

def to_confusion_matrix(truth, predictions, classes):
    t = np.select([truth==i for i in np.unique(truth)],classes, truth)
    p = np.select([predictions==i for i in np.unique(predictions)],classes, predictions)


    cm = confusion_matrix(t, p, labels=np.array(classes))
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()

# get accuracy at patch level
def truth(labels, predictions, classes):
    l = labels[:, 0]
    print(classification_report(l, predictions, target_names=classes, digits=4))

    to_confusion_matrix(l, predictions, classes)



def main(model_name, input1, input2, classes):
    model = models.load_model(model_name)

    # predictions represented as integer representation of classes
    predictions = get_predictions(input, classes, model)


    # labels with class and image number
    labels = get_labels(input)


    
    truth(labels, predictions, classes)

   

