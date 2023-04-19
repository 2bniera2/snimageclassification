from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages

import importlib

from utils.data_generator import data_generator
from utils.time_callback import time_logger
from keras import callbacks
from utils.plot_acc_loss import plot_acc_loss



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
        patience=10,
        # restore_best_weights=True,
        # verbose=1
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




