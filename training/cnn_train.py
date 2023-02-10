'''
Usage: python cnn_train.py {name} {EPOCH} {BATCH}
'''

from sys import argv, path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
from tensorflow.keras import models, callbacks
from sklearn import utils
import numpy as np
import h5py
import model_architectures
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# obtain dataset lengths
# def get_dset_len(path, dset):
#     with h5py.File(path, 'r') as f:
#         return f[dset].shape[0]

# def one_hot_encode(label):
#     label_map = {
#         'facebook': 0,
#         'flickr': 1,
#         'google+': 2,
#         'instagram': 3,
#         'original': 4,
#         'telegram': 5,
#         'twitter': 6,
#         'whatsapp': 7
#     }
#     one_hot_label = np.zeros(len(label_map))
#     one_hot_label[label_map[label.decode('UTF-8')]] = 1
#     return one_hot_label


# # generator function to feed data to model in batches
# def generator(batch_size, num_examples, task, name, shuffle):

#     with h5py.File(f'{path[0]}/processed/DCT_{task}_{name}.h5', 'r') as Examples, h5py.File(f'{path[0]}/processed/labels_{task}_{name}.h5', 'r') as Labels:
#         D_X = Examples['DCT']
#         D_y = Labels['labels']


#         while True:
#             i = 0
#             Examples_batch = []
#             Labels_batch = []
#             indices = [i for i in range(num_examples)]

#             if shuffle:
#                 random.shuffle(indices)


#             while indices:
#                 while i < batch_size and indices:
#                     idx = indices.pop()
#                     Examples_batch.append(D_X[idx])
#                     Labels_batch.append(one_hot_encode(D_y[idx]))
#                     i += 1
#                 yield (np.array(Examples_batch), np.array(Labels_batch))




def main(name, epoch, batch_size, architecture, his_range, sf_range):



    # train_dset_len = get_dset_len(f'{path[0]}/processed/DCT_train_{name}.h5', 'DCT')
    # val_dset_len =  get_dset_len(f'{path[0]}/processed/DCT_val_{name}.h5', 'DCT')

    with h5py.File(f'{path[0]}/processed/DCT_train_{name}.h5') as train, h5py.File(f'{path[0]}/processed/DCT_val_{name}.h5') as val:
        examples_train = np.array(train['DCT'][:]) 
        examples_val = np.array(val['DCT'][:])

    with h5py.File(f'{path[0]}/processed/labels_train_{name}.h5') as train, h5py.File(f'{path[0]}/processed/labels_val_{name}.h5') as val:
        labels_train = [i.decode('UTF-8') for i in np.array(train['labels'][:])]
        labels_val = [i.decode('UTF-8') for i in np.array(val['labels'][:])]


    encoder = LabelEncoder()   
    labels_train = encoder.fit_transform(labels_train)
    labels_val = encoder.fit_transform(labels_val)


    labels_train = to_categorical(labels_train)
    labels_val = to_categorical(labels_val)



    # print(labels_train)

    # layers
    model = models.Sequential()


    input_shape = ((his_range[1]*2 + 1) * sf_range, 1)


    getattr(model_architectures, architecture)(model, input_shape)
    
    # early stopping to prevent doing unnecessary epochs
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # train
    history = model.fit(
        examples_train,
        labels_train,
        epochs=epoch,
        batch_size=batch_size,
        callbacks=[callback],
        validation_data=(examples_val, labels_val),
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # save
    model.save(f'models/cnn_{argv[1]}')

if __name__ == "__main__":
    main()
