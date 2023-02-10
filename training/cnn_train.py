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


# obtain dataset lengths
def get_dset_len(path, dset):
    with h5py.File(path, 'r') as f:
        return f[dset].shape[0]

def one_hot_encode(label):
    label_map = {
        'facebook': 0,
        'flickr': 1,
        'google+': 2,
        'instagram': 3,
        'original': 4,
        'telegram': 5,
        'twitter': 6,
        'whatsapp': 7
    }
    one_hot_label = np.zeros(len(label_map))
    one_hot_label[label_map[label.decode('UTF-8')]] = 1
    return one_hot_label


# generator function to feed data to model in batches
def generator(batch_size, num_examples, task, name, shuffle):

    with h5py.File(f'{path[0]}/processed/DCT_{task}_{name}.h5', 'r') as Examples, h5py.File(f'{path[0]}/processed/labels_{task}_{name}.h5', 'r') as Labels:
        D_X = Examples['DCT']
        D_y = Labels['labels']

        length = D_X.shape[0]

        while True:
            indices = [i for i in range(0, num_examples, batch_size)]
            if shuffle:
                indices = np.random.permutation(indices)
            for i in indices:
                # prevents exceeding bounds of list
                Examples_batch = D_X[i: min(i + batch_size, num_examples)]
                Labels_batch = D_y[i: min(i + batch_size, num_examples)]

                one_hot_labels = []
                # one hot encode labels
                for label in Labels_batch:
                    one_hot_labels.append(one_hot_encode(label))

                yield (np.array(Examples_batch), np.array(one_hot_labels))




def main(name, epoch, batch_size, architecture, his_range, sf_range):

    train_dset_len = get_dset_len(f'{path[0]}/processed/DCT_train_{name}.h5', 'DCT')
    val_dset_len =  get_dset_len(f'{path[0]}/processed/DCT_val_{name}.h5', 'DCT')

    # layers
    model = models.Sequential()


    input_shape = ((his_range[1]*2 + 1) * sf_range, 1)

    print(input_shape)

    getattr(model_architectures, architecture)(model, input_shape)
    
    # early stopping to prevent doing unnecessary epochs
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # train
    history = model.fit(
        generator(batch_size, train_dset_len, 'train', name, True),
        steps_per_epoch=np.ceil(train_dset_len / batch_size),
        epochs=epoch,
        callbacks=[callback],
        validation_data=(generator(batch_size, val_dset_len, 'val', name, False)),
        validation_steps=np.ceil(val_dset_len / batch_size),
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
