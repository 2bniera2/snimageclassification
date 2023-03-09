from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages

import importlib

from data_generator import data_generator
from keras import callbacks




def main(epochs, batch_size, architecture, location, input, classes, name):
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
        restore_best_weights=True
    )


    csv_logger = callbacks.CSVLogger(f'{name}.log')

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[csv_logger, earlystop],
        use_multiprocessing=True,
        workers=6
    )
    model.save(name)
