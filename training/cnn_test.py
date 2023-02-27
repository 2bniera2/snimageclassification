import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sys import path
import h5py
from data_generator import data_generator


def make_name(architecture, input_shape, epochs, batch_size):
    return f'models/cnn_{architecture}_{input_shape}_{epochs}_{batch_size}'

def get_labels(input):
    with h5py.File(f'processed/{input.domain}_test_{input.dset_name}.h5', 'r') as f:
        return np.array(f['labels'][()])


# get predictions and convert numerical values to class name
def get_predictions(input, model):
    gen = data_generator(
        f'{path[0]}/processed/{input.domain}_test_{input.dset_name}.h5',
        input.domain,
        input.classes,
        shuffle=False
    )

    pred = model.predict(gen, use_multiprocessing=True, workers=8)
    return pred, np.argmax(pred, axis=1)

def to_confusion_matrix(truth, predictions, classes):
    t = np.select([truth==i for i in np.unique(truth)],classes, truth)
    p = np.select([predictions==i for i in np.unique(predictions)],classes, predictions)


    cm = confusion_matrix(t, p, labels=np.array(classes))
    print(cm)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # disp.plot()
    # plt.show()




# get accuracy at patch level
def patch_truth(labels, predictions, classes):
    l = labels[:, 0]
    print(classification_report(l, predictions, target_names=classes, digits=4))

    to_confusion_matrix(l, predictions, classes)

# get accuracy at image level
def image_truth(labels, predictions, classes):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions], index=['truth', 'image_num', 'prediction']).T
    df = df.groupby(['truth','image_num'])['prediction'].agg(pd.Series.mode).reset_index()
    df = df[pd.notna(pd.to_numeric(df['prediction'], errors='coerce'))]
    df = df.reset_index().drop('image_num', axis=1)

    image_truth = df['truth'].to_numpy().astype(np.uint8)
    image_predictions = df['prediction'].to_numpy().astype(np.uint8)

    print(classification_report(image_truth, image_predictions, target_names=classes, digits=4))

    to_confusion_matrix(image_truth, image_predictions, classes)


def main(input, epochs, batch_size, architecture):
    model = models.load_model(make_name(architecture, input.input_shape, epochs, batch_size))

    # predictions represented as integer representation of classes
    probs, predictions = get_predictions(input, model)

    # labels with class and image number
    labels = get_labels(input)

    patch_truth(labels, predictions, input.classes)
    image_truth(labels, predictions, input.classes)

if __name__ == "__main__":
    main()