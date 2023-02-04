'''
Usage: python cnn_test.py {file to load} {save results}
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
from sys import argv
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np


def return_dset_len(path, dset):
    with h5py.File(path, 'r') as f:
        return f[dset].shape[0]


# generator function
def generator(name, batch_size, num_examples):
    with h5py.File(f'processed/DCT_test_{name}.h5', 'r') as f:
        X = f['DCT']
        while True:
            for i in range(0, num_examples, batch_size):
                batch_X = X[i: min(i + batch_size, num_examples)]
                yield (batch_X)

def get_labels(name):
    with h5py.File(f'processed/labels_test_{name}.h5', 'r') as f:
        return np.array(f['labels'][()])

def get_predictions(name, model, num_examples):
    predictions = np.argmax(model.predict(generator(name, 5000, num_examples), steps=np.ceil(num_examples/5000)), axis=1)  

    return np.select(
        [
            predictions == 0,
            predictions == 1,
            predictions == 2,
            predictions == 3,
            predictions == 4,
            predictions == 5,
            predictions == 6,
            predictions == 7,

        ],
        [
            'facebook',
            'flickr',
            'google+',
            'instagram',
            'original',
            'telegram',
            'twitter',
            'whatsapp'
        ],
        predictions
    )



def patch_truth(labels, predictions, classes):
        
    # class_int_map = {
    #     'facebook': 0,
    #     'flickr': 1,
    #     'google+': 2,
    #     'instagram': 3,
    #     'original': 4,
    #     'telegram': 5,
    #     'twitter': 6,
    #     'whatsapp': 7
    # }

    
    patch_truth = [label.decode('UTF-8').split('.')[0] for labels in label]

    # # convert labels with image index to just regular labels (to test on patch level)
    # for label in labels:
    #     l = label.decode('UTF-8')
    #     truth = l.split('.')
    #     patch_truth.append(truth[0])

    print(classification_report(patch_truth, predictions, target_names=classes))


# def image_truth(name, predictions, classes):

def image_truth(labels, predictions, classes):
    #decode
    y_test_im = []
    for y in labels:
        y_test_im.append(y.decode('UTF-8'))

    df = pd.DataFrame([y_test_im, predictions], index=['truth', 'prediction']).T
    # group by class and image number
    grouped_df = df.groupby('truth', as_index=False)[
        'prediction'].agg(pd.Series.mode)

    # split into respective image number
    grouped_df['truth'] = grouped_df['truth'].str.split('.').str[0]   
    print(grouped_df)

    image_truth = grouped_df['truth'].to_numpy()
    image_prediction = grouped_df['prediction'].to_numpy()



    # print(image_truth)
    # print(image_prediction)
    # y_truth = np.select([
    #     y_truth == 'facebook',
    #     y_truth == 'flickr',
    #     y_truth == 'google+',
    #     y_truth == 'instagram',
    #     y_truth == 'original',
    #     y_truth == 'telegram',
    #     y_truth == 'twitter',
    #     y_truth == 'whatsapp'
    # ], [0,1,2,3,4,5,6,7], y_truth).astype(np.uint8)


    # y_pred = np.select([
    #     y_pred == 'facebook',
    #     y_pred == 'flickr',
    #     y_pred == 'google+',
    #     y_pred == 'instagram',
    #     y_pred == 'original',
    #     y_pred == 'telegram',
    #     y_pred == 'twitter',
    #     y_pred == 'whatsapp'
    # ], [0,1,2,3,4,5,6,7], y_pred).astype(np.uint8)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # user defined variables
    name = argv[1]
    results = argv[2]

    model = models.load_model(f'models/cnn_{argv[1]}')

    classes = ['facebook', 'flickr', 'google+', 'instagram',
            'original', 'telegram', 'twitter', 'whatsapp']

    # get the number of examples for the generator and steps
    num_examples = return_dset_len(f'processed/DCT_test_{name}.h5', 'DCT')

    # predictions represented as integer representation of classes
    predictions = get_predictions(name, model, num_examples)

    # labels with string name and image indexes
    labels = get_labels(name)

    patch_truth(labels, predictions, classes)

    image_truth(labels, predictions, classes)

























# # print(classification_report(y_truth, y_pred, target_names=classes))

# # cm = confusion_matrix(
# # 	y_truth,
# # 	y_pred,
# # 	labels=classes,
# #     normalize='true'

# # )

# # cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
# # fig, ax = plt.subplots(figsize=(10, 10))
# # cm_display.plot(ax=ax)
# # plt.savefig(f'results/{argv[2]}_whole.png')


if __name__ == "__main__":
    
    main()