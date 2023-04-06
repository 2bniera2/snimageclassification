from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages

import importlib

from data_generator import data_generator
from time_callback import time_logger
from keras import callbacks
from matplotlib import pyplot as plt

def plot_acc_loss(history, name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'training_accuracy/{name}.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'training_loss/{name}.png')
    plt.show()





def train(epochs, batch_size, architecture, location, input, classes, name):
    train_gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_train.h5',
        'examples',
        classes,
        batch_size
    )
    val_gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_val.h5',
        'examples',
        classes,
        batch_size,
        shuffle=False
    )

    module = importlib.import_module(location)
    model = getattr(module, architecture)(input.input_shape, len(classes))

    earlystop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=0,
        restore_best_weights=True,
        verbose=1
    )


    csv_logger = callbacks.CSVLogger(f'logs/{name}.csv')

    time_log = time_logger(f'train_times/{name}.csv')

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[csv_logger, earlystop, time_log],
        use_multiprocessing=True,
        workers=16,
    )

    plot_acc_loss(history, name)

    model.save(f'models/{name}')




