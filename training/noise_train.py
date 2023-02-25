from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import model_architectures
from data_generator import data_generator


def main(name, dataset_name, epoch, batch_size, architecture, input):
    train_gen = data_generator(
        f'{path[0]}/processed/Noise_train_{dataset_name}.h5',
        'Noise',
        batch_size,
    )
    val_gen = data_generator(
        f'{path[0]}/processed/Noise_val_{dataset_name}.h5',
        'Noise',
        batch_size
    )

    input_shape = (input.patch_size, input.patch_size, 1)

    model = getattr(model_architectures, architecture)(input_shape)

    # train
    history = model.fit(
        train_gen,
        epochs = epoch,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=6
    )
    
    model.save(f'models/cnn_noise_{name}')
