from sys import path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
import multi_input_cnn.multi_input_models as multi_input_models

from utils.multi_input_data_generator import multi_input_data_generator
from keras import callbacks, layers, models

def FusionNET(input1_shape, input2_shape, output_shape):

    #dct
    in1 = layers.Input(shape=input1_shape)
    conv1 = layers.Conv1D(100, 3, activation='relu')(in1)
    max_pool1 = layers.MaxPooling1D()(conv1)
    conv1 = layers.Conv1D(100, 3, activation='relu')(max_pool1)
    max_pool1 = layers.MaxPooling1D()(conv1)
    dense1 = layers.Dense(256, activation='relu')(max_pool1)
    flat1 = layers.Flatten()(dense1)
    f1 = layers.Dropout(rate=0.5)(flat1)

    #noise
    in2 = layers.Input(shape=input2_shape)
    conv2 = layers.Conv2D(32, (3,3), activation='relu')(in2)
    conv2 = layers.Conv2D(32, (3,3), activation='relu')(conv2)
    max_pool2 = layers.MaxPool2D()(conv2)
    dropout2 = layers.Dropout(rate=0.5)(max_pool2)
    conv2 = layers.Conv2D(64, (3,3), activation='relu')(dropout2)
    conv2 = layers.Conv2D(64, (3,3), activation='relu')(conv2)
    max_pool2 = layers.MaxPool2D()(conv2)
    dropout2 = layers.Dropout(rate=0.5)(max_pool2)
    flat2 = layers.Flatten()(dropout2)
    dense2 = layers.Dense(256, activation='relu')(flat2)
    f2 = layers.Dropout(rate=0.5)(dense2)

    concat = layers.Concatenate()([f1, f2])

    dense = layers.Dense(512, activation='relu')(concat)
    output = layers.Dense(units=output_shape, activation='softmax')(dense)

    model = models.Model(inputs=[in1, in2], outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model

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
        f'{path[0]}/processed/{n_input.dset_name}_val.h5',
        'examples',
        'examples',
        classes,
        batch_size
    )
   
    model = FusionNET(h_input.input_shape, n_input.input_shape, len(classes))

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