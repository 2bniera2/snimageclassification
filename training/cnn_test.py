import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages
import numpy as np
from keras import models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sys import path
import h5py
from data_generator import data_generator


def get_labels(dataset_name):
    with h5py.File(f'processed/DCT_test_{dataset_name}.h5', 'r') as f:
        return np.array(f['labels'][()])


# get predictions and convert numerical values to class name
def get_predictions(dataset_name, model):
    gen = data_generator(
        f'{path[0]}/processed/DCT_test_{dataset_name}.h5',
        'DCT',
        32,
        shuffle=False
    )
    return np.argmax(model.predict(
        gen, use_multiprocessing=True, workers=8), axis=1)


# get accuracy at patch level
def patch_truth(labels, predictions, classes):
    print(classification_report(labels[:, 0], predictions, target_names=classes, digits=4))


# get accuracy at image level
def image_truth(labels, predictions, classes):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions], index=['truth', 'image_num', 'prediction']).T
    df = df.groupby(['truth','image_num'])['prediction'].agg(pd.Series.mode).reset_index()
    df = df[pd.notna(pd.to_numeric(df['prediction'], errors='coerce'))]
    df = df.reset_index().drop('image_num', axis=1)

    image_truth = df['truth'].tolist()
    image_predictions = df['prediction'].tolist()

    print(classification_report(image_truth, image_predictions, target_names=classes, digits=4))


def test(name, dataset_name):
    model = models.load_model(f'models/cnn_{name}')

    classes = ['facebook', 'flickr', 'google+', 'instagram',
               'original', 'telegram', 'twitter', 'whatsapp']

    # predictions represented as integer representation of classes
    predictions = get_predictions(dataset_name, model)

    # labels with class and image number
    labels = get_labels(dataset_name)

    patch_truth(labels, predictions, classes)

    image_truth(labels, predictions, classes)
