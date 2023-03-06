from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import training.multi_input_models as multi_input_models

from data_generator import data_generator
from keras import callbacks


def main(epochs, batch_size, architecture, h_input, n_input, classes, name):
    dct_train_gen = data_generator(
        f'{path[0]}/processed/{h_input.dset_name}_train.h5',
        'examples',
        classes,
        batch_size
    )
    dct_val_gen = data_generator(
        f'{path[0]}/processed/{h_input.dset_name}_val.h5',
        'examples',
        classes,
        batch_size
    )

    noise_train_gen = data_generator(
        f'{path[0]}/processed/{n_input.dset_name}_train.h5',
        'examples',
        classes,
        batch_size
    )
    noise_val_gen = data_generator(
        f'{path[0]}/processed/{n_input.dset_name}_val.h5',
        'examples',
        classes,
        batch_size
    )
    

    model = getattr(multi_input_models, architecture)(h_input.input_shape, n_input.input_shape, len(classes))

    earlystop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )


    csv_logger = callbacks.CSVLogger(f'{name}.log')

    history = model.fit(
        [dct_train_gen, noise_train_gen],
        epochs=epochs,
        validation_data=[dct_val_gen, noise_val_gen],
        callbacks=[earlystop, csv_logger],
        use_multiprocessing=True,
        workers=6
    )
    model.save(name)