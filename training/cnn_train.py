from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import model_architectures
from data_generator import data_generator


def main(epochs, batch_size, architecture, input):
    train_gen = data_generator(
        f'{path[0]}/processed/{input.domain}_train_{input.dset_name}.h5',
        input.domain,
        batch_size,
    )
    val_gen = data_generator(
        f'{path[0]}/processed/{input.domain}_val_{input.dset_name}.h5',
        input.domain,
        batch_size
    )

    model = getattr(model_architectures, architecture)(input.input_shape)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=6
    )
    model.save(f'models/cnn_{architecture}_e:{epochs}_b:{batch_size}')
