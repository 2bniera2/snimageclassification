import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages
import numpy as np
from keras import models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
import pandas as pd
from sys import path
import h5py
from data_generator import data_generator



# get the length for the dataset, used for generator function and to calculate steps_per_epoch
def get_dset_len(path, dset):
    with h5py.File(path, 'r') as f:
        return f[dset].shape[0]


def get_labels(dataset_name):
    with h5py.File(f'processed/DCT_test_{dataset_name}.h5', 'r') as f:
        return np.array(f['labels'][()])


# get predictions and convert numerical values to class name
def get_predictions(dataset_name, model, num_examples):
    gen = data_generator(
        f'{path[0]}/processed/DCT_test_{dataset_name}.h5',
        'DCT',
        num_examples,
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
    # df organised by ground truth, image number and the prediction
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions],
                      index=['truth', 'image_num', 'prediction']).T

    # group by ground truth and image number and aggregate on the mode of prediction
    df = df.groupby(['truth','image_num'])['prediction'].agg(pd.Series.mode).reset_index()

    df = df[pd.notna(pd.to_numeric(df['prediction'], errors='coerce'))]
    df = df.reset_index().drop('image_num', axis=1)
    image_truth = df['truth'].tolist()
    image_predictions = df['prediction'].tolist()

    print(classification_report(image_truth, image_predictions, target_names=classes, digits=4))


def main(name, dataset_name):
    model = models.load_model(f'models/cnn_{name}')

    classes = ['facebook', 'flickr', 'google+', 'instagram',
               'original', 'telegram', 'twitter', 'whatsapp']

    # get the number of examples for the generator and steps
    num_examples = get_dset_len(f'{path[0]}/processed/DCT_test_{dataset_name}.h5', 'DCT')

    # predictions represented as integer representation of classes
    predictions = get_predictions(dataset_name, model, num_examples)

    # labels with class and image number
    labels = get_labels(dataset_name)

    patch_truth(labels, predictions, classes)

    image_truth(labels, predictions, classes)


if __name__ == "__main__":
    main()
