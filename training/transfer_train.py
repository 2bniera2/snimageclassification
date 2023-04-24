from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
from utils.regulariser import add_regularization
import importlib

path.append(f'{os.getcwd()}/utils')
from utils.data_generator import data_generator
from utils.time_callback import time_logger
from keras import callbacks, layers, models, applications, regularizers, optimizers
from utils.plot_acc_loss import plot_acc_loss


def vgg16(input_shape, output_shape, model_input):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    

    base = applications.VGG16(include_top=False,input_shape=input_shape, weights='imagenet')(input)

    return compile_model(base, input, output_shape)


def resnet50(input_shape, output_shape, model_input):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    preprocess = applications.resnet.preprocess_input(input)
    base = applications.ResNet50(include_top=False,input_shape=input_shape, weights='imagenet')(preprocess)

    return compile_model(base, input, output_shape)

# compile model with optimiser, as well as deciding if to freeze weights and add regularisation
def compile_model(base, input, output_shape):
    flat = layers.Flatten()(base)
    dense1 = layers.Dense(4096, activation="relu")(flat)

    output = layers.Dense(output_shape, activation="softmax")(dense1)

    model = models.Model(inputs=input, outputs=output)

    model = add_regularization(model, regularizers.l2(0.1))


    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(0.00001),
        metrics=['accuracy']
    )

    # for layer in model.layers[:-3]:
    #     layer.trainable = False

    model.summary()

    return model

# main training function that initialises generators and calls the model fitting 
def main(input, model_input, classes, name):

    # generator functions 
    train_gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_train.h5',
        'examples',
        classes,
        model_input.batch_size
    )
    val_gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_val.h5',
        'examples',
        classes,
        model_input.batch_size
    )

    # get model architecture
    module = importlib.import_module('transfer_train')
    model = getattr(module, model_input.architecture)(input.input_shape, len(classes), model_input)

    # callbacks for logging and early stopping
    csv_logger = callbacks.CSVLogger(f'logs/{name}.csv')
    earlystop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    time_log = time_logger(f'train_times/{name}.csv')


    # fit frozen model with unfrozen classification layers
    history = model.fit(
        train_gen,
        epochs=model_input.epochs,
        validation_data=val_gen,
        callbacks=[csv_logger, earlystop, time_log],
        use_multiprocessing=True,
        workers=16
    )

    # save frozen model training and validation accuracy and loss
    # plot_acc_loss(history, f'{name}_frozen')



    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(0.00001),
        metrics=['accuracy']
    )
    print(model.losses)

    # # fit unfrozen model (fine tuning)
    for layer in model.layers[:-2]:
        layer.trainable = True

    model.summary()

    history = model.fit(
        train_gen,
        epochs=model_input.epochs,
        validation_data=val_gen,
        callbacks=[csv_logger, earlystop, time_log],
        use_multiprocessing=True,
        workers=16
    )

    # save finetuned model training and validation accuracy and loss
    # plot_acc_loss(history, f'{name}_finetuned')


    # save the model with all its trained weights
    model.save(f'models/{name}')
