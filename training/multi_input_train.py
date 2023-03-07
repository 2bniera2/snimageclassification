from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import training.multi_input_models as multi_input_models

from multi_input_data_generator import multi_input_data_generator
from keras import callbacks


def main(epochs, batch_size, architecture, h_input, n_input, classes, name):
    train_gen = multi_input_data_generator(
        f'{path[0]}/processed/{h_input.dset_name}_train.h5',
        f'{path[0]}/processed/{n_input.dset_name}_train.h5',
        'examples',
        'examples',
        classes,
        batch_size
    )
    val_gen = multi_input_data_generator(
        f'{path[0]}/processed/{h_input.dset_name}_val.h5',
        f'{path[0]}/processed/{n_input.dset_name}_train.h5',
        'examples',
        'examples',
        classes,
        batch_size
    )
    print(h_input.input_shape)
    print(n_input.input_shape)


    model = getattr(multi_input_models, architecture)(h_input.input_shape, n_input.input_shape, len(classes))

    earlystop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )


    csv_logger = callbacks.CSVLogger(f'{name}.log')

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[earlystop, csv_logger],
        use_multiprocessing=True,
        workers=6
    )
    model.save(name)