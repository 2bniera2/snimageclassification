from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages

import importlib

path.append(f'{os.getcwd()}/utils')
from utils.data_generator import data_generator
from utils.time_callback import time_logger
from keras import callbacks, layers, models, applications, regularizers
from utils.regulariser import add_regularization
from utils.plot_acc_loss import plot_acc_loss


def vgg16(input_shape, output_shape, model_input):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    
    base = applications.VGG16(include_top=False,input_shape=input_shape, weights=model_input.weights)(input)
    return compile_model(base, input, output_shape, model_input)

def vgg19(input_shape, output_shape, model_input):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    
    base = applications.VGG19(include_top=False,input_shape=input_shape, weights=model_input.weights)(input)
    return compile_model(base, input, output_shape, model_input)

def resnet50(input_shape, output_shape, model_input):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    
    base = applications.ResNet50(include_top=False,input_shape=input_shape, weights=model_input.weights)(input)

    return compile_model(base, input, output_shape, model_input)


def inception(input_shape, output_shape, model_input):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    
    base = applications.InceptionV3(include_top=False,input_shape=input_shape, weights=model_input.weights)(input)
   
    return compile_model(base, input, output_shape, model_input)


# compile model with optimiser, as well as deciding if to freeze weights and add regularisation
def compile_model(base, input, output_shape, model_input):
    if not model_input.trainable: base.trainable = False
    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    if model_input.regularize: model = add_regularization(model, regularizers.l2(0.001))


    model.compile(
        loss='categorical_crossentropy',
        optimizer=model_input.optimizer,
        metrics=['accuracy']
    )

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
    module = importlib.import_module('_2d_train')
    model = getattr(module, model_input.architecture)(input.input_shape, len(classes), model_input)

    # callbacks for logging and early stopping
    csv_logger = callbacks.CSVLogger(f'logs/{name}.csv')
    earlystop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    time_log = time_logger(f'train_times/{name}.csv')

    # fit model
    history = model.fit(
        train_gen,
        epochs=model_input.epochs,
        validation_data=val_gen,
        callbacks=[csv_logger, earlystop, time_log],
        use_multiprocessing=True,
        workers=16
    )

    # save model training and validation accuracy and loss
    plot_acc_loss(history, name)

    # save the model with all its trained weights
    model.save(f'models/{name}')
