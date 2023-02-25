from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import model_architectures
from data_generator import data_generator


def train(name, epoch, batch_size, architecture, input):
    train_gen = data_generator(
        f'{path[0]}/processed/DCT_train_{input.dset_name}.h5',
        'DCT',
        batch_size,
    )
    val_gen = data_generator(
        f'{path[0]}/processed/DCT_val_{input.dset_name}.h5',
        'DCT',
        batch_size
    )

    input_shape = (input.his_size, 1)

    model = getattr(model_architectures, architecture)(input_shape)

    # train
    history = model.fit(
        train_gen,
        epochs = epoch,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=6
    )
    model.save(f'models/cnn_{name}')
