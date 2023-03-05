from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import training.dct_models as dct_models
import training.dct_models as dct_models
import training.dct_models as dct_models

from data_generator import data_generator
from keras import callbacks




def main(epochs, batch_size, architecture, input, classes, name):
    train_gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_train.h5',
        input.domain,
        classes,
        batch_size
    )
    val_gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_val.h5',
        input.domain,
        classes,
        batch_size
    )

    model = getattr(dct_models, architecture)(input.input_shape, len(classes))

    callback = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[callback],
        use_multiprocessing=True,
        workers=6
    )
    model.save(name)
