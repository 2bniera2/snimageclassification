from tensorflow.keras import layers, models


def dct_cnn_2017(input_shape):
    input = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(100, 3, activation='relu', padding="valid")(input)
    max_pool1 = layers.MaxPooling1D()(conv1)
    conv2 = layers.Conv1D(100, 3, activation='relu', padding="valid")(max_pool1)
    max_pool2 = layers.MaxPooling1D()(conv2)
    flat = layers.Flatten()(max_pool2)
    dense1 = layers.Dense(units=256, activation='relu')(flat)
    dense2 =  layers.Dense(units=256)(dense1)
    dropout = layers.Dropout(rate=0.5)(dense2)
    output = layers.Dense(units=8, activation='softmax')(dropout)
    model = models.Model(inputs=input, outputs=output) 
    # model.add(layers.Conv1D(100, 3, activation='relu', padding="valid", input_shape=input_shape))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Conv1D(100, 3, activation='relu', padding="valid"))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=256, activation='relu'))
    # model.add(layers.Dense(units=256))
    # model.add(layers.Dropout(rate=0.5))
    # model.add(layers.Dense(units=8, activation='softmax'))


    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model


def dct_cnn_2019(input_shape):
    input = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(100, 3, activation='relu', padding="valid")(input)
    max_pool1 = layers.MaxPooling1D()(conv1)
    conv2 = layers.Conv1D(100, 3, activation='relu', padding="valid")(max_pool1)
    max_pool2 = layers.MaxPooling1D()(conv2)
    flat = layers.Flatten()(max_pool2)
    dense1 = layers.Dense(units=1000, activation='relu')(flat)
    dropout1 = layers.Dropout(rate=0.5)(dense1)
    dense2 =  layers.Dense(units=1000)(dropout1)
    dropout2 = layers.Dropout(rate=0.5)(dense2)
    output = layers.Dense(units=8, activation='softmax')(dropout2)
    model = models.Model(inputs=input, outputs=output) 

    # model.add(layers.Conv1D(100, 3, activation='relu', input_shape=input_shape))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Conv1D(100, 3, activation='relu'))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=1000))
    # model.add(layers.Dropout(rate=0.5))
    # model.add(layers.Dense(units=1000))
    # model.add(layers.Dropout(rate=0.5))
    # model.add(layers.Dense(units=8, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model

