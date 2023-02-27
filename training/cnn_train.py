from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import model_architectures
from data_generator import data_generator
from keras import callbacks


def make_name(architecture, input_shape, epochs, batch_size):
    return f'models/cnn_{architecture}_{input_shape}_{epochs}_{batch_size}'


def main(epochs, batch_size, architecture, input):
    train_gen = data_generator(
        f'{path[0]}/processed/{input.domain}_train_{input.dset_name}.h5',
        input.domain,
        input.classes,
        batch_size
    )
    val_gen = data_generator(
        f'{path[0]}/processed/{input.domain}_val_{input.dset_name}.h5',
        input.domain,
        input.classes,
        batch_size
    )

    model = getattr(model_architectures, architecture)(input.input_shape, len(input.classes))

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
    model.save(make_name(architecture, input.input_shape, epochs, batch_size))
