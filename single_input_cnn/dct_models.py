from keras import layers, models

def dct_cnn_2017(input_shape, output_shape):
    input = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(100, 3, activation='relu')(input)
    max_pool1 = layers.MaxPooling1D()(conv1)
    conv2 = layers.Conv1D(100, 3, activation='relu')(max_pool1)
    max_pool2 = layers.MaxPooling1D()(conv2)
    flat = layers.Flatten()(max_pool2)
    dense1 = layers.Dense(units=256, activation='relu')(flat)
    dropout1 = layers.Dropout(rate=0.5)(dense1)
    dense2 =  layers.Dense(units=256, activation='relu')(dropout1)
    dropout2 = layers.Dropout(rate=0.5)(dense2)
    output = layers.Dense(units=output_shape, activation='softmax')(dropout2)
    model = models.Model(inputs=input, outputs=output) 

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model


def dct_cnn_2019(input_shape, output_shape):
    input = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(100, 3, activation='relu')(input)
    max_pool1 = layers.MaxPooling1D()(conv1)
    conv2 = layers.Conv1D(100, 3, activation='relu')(max_pool1)
    max_pool2 = layers.MaxPooling1D()(conv2)
    flat = layers.Flatten()(max_pool2)
    dense1 = layers.Dense(units=1000, activation='relu')(flat)
    dropout1 = layers.Dropout(rate=0.5)(dense1)
    dense2 =  layers.Dense(units=1000)(dropout1)
    dropout2 = layers.Dropout(rate=0.5)(dense2)
    output = layers.Dense(units=output_shape, activation='softmax')(dropout2)
    model = models.Model(inputs=input, outputs=output) 

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model

def dct_cnn2(input_shape, output_shape):
    input_shape = (20,11,9)
    input = layers.Input(shape=input_shape)
    conv = layers.Conv2D(64, (3,3), activation='relu')(input)
    batch1 = layers.BatchNormalization()(conv)
    conv1 = layers.Conv2D(64, (3,3), activation='relu')(batch1)
    maxpool = layers.MaxPooling2D()(conv1)
    batch2 = layers.BatchNormalization()(maxpool)
    conv2 = layers.Conv2D(128, (3,3), activation='relu')(batch2)
    batch3 = layers.BatchNormalization()(conv2)
    conv3 = layers.Conv2D(128, (3,3), activation='relu')(batch3)
    flat = layers.Flatten()(conv3)
    dense1 = layers.Dense(256, activation='swish')(flat)
    dropout1 = layers.Dropout(rate=0.8)(dense1)
    dense2 = layers.Dense(256, activation='swish')(dropout1)
    dropout2 = layers.Dropout(rate=0.8)(dense2)
    dense3 = layers.Dense(256, activation='swish')(dropout2)
    dropout3 = layers.Dropout(rate=0.8)(dense3)
    dense4 = layers.Dense(256, activation='swish')(dropout3)
    output = layers.Dense(units=output_shape, activation='softmax')(dense4)
    model = models.Model(inputs=input, outputs=output) 

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Nadam',
        metrics=['accuracy']
    )

    return model


