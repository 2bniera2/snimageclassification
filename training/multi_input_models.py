from keras import layers, models


def FusionNET(input_shape, output_shape):
    #dct
    in1 = layers.Input(shape=input)
    conv = layers.Conv1D(100, 3, activation='relu')(in1)
    max_pool = layers.MaxPooling1D()(conv1)
    conv = layers.Conv1D(100, 3, activation='relu')(max_pool)
    max_pool = layers.MAxPooling1D()(conv)
    dense = layers.Dense(256, activation='relu')(max_pool)
    flat = layers.Flatten()(dense/)
    f1 = layers.Dropout(rate=0.5)(flat)

    #noise
    in2 = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3,3), activation='relu')(input)
    conv2 = layers.Conv2D(32, (3,3), activation='relu')(conv1)
    max_pool1 = layers.MaxPool2D()(conv2)
    dropout1 = layers.Dropout(rate=0.5)(max_pool1)
    conv3 = layers.Conv2D(64, (3,3), activation='relu')(dropout1)
    conv4 = layers.Conv2D(64, (3,3), activation='relu')(conv3)
    max_pool2 = layers.MaxPool2D()(conv4)
    dropout2 = layers.Dropout(rate=0.5)(max_pool2)
    flat = layers.Flatten()(dropout2)
    dense = layers.Dense(256, activation='relu')(flat)
    f2 = layers.Dropout(rate=0.5)(dense)

    concat = layers.Concatenate()([f1, f2])

    dense = layers.Dense(512, activation='relu')(concat)
    output = layers.Dense(units=output_shape, activation='softmax')(dense)

    model = models.Model(inputs=[in1, in2], output=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model

def two_stream(input_shape, output_shape):
    #dct
    input1 = layers.Input(shape=input_shape)
    conv = layers.Conv2D(64, (3,3), activation='relu')(input1)
    batch = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(64, (3,3), activation='relu')(batch)
    maxpool = layers.MaxPooling2D()(conv)
    batch = layers.BatchNormalization()(maxpool)
    conv = layers.Conv2D(128, (3,3), activation='relu')(batch)
    batch = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(128, (3,3), activation='relu')(batch)
    f1 = layers.Flatten()(conv)
    #noise
    input2 = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(16, (3,3), activation='relu')(input2)
    max_pool1 = layers.MaxPooling2D()(conv1)
    conv2 = layers.Conv2D(32, (3,3), activation='relu')(max_pool1)
    max_pool2 = layers.MaxPooling2D()(conv2)
    conv3 = layers.Conv2D(16, (3,3), activation='relu')(max_pool2)
    f2 = layers.Flatten()(conv3)

    concat = layers.Concatenate()([f1, f2])

    dense1 = layers.Dense(256, activation='swish')(concat)
    dropout1 = layers.Dropout(rate=0.8)(dense1)
    dense2 = layers.Dense(256, activation='swish')(dropout1)
    dropout2 = layers.Dropout(rate=0.8)(dense2)
    dense3 = layers.Dense(256, activation='swish')(dropout2)
    dropout3 = layers.Dropout(rate=0.8)(dense3)
    dense4 = layers.Dense(256, activation='swish')(dropout3)
    output = layers.Dense(units=output_shape, activation='softmax')(dense4)

    model = models.Model(inputs=[input1, input2], outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Nadam',
        metrics=['accuracy']
    )

    return model

def msf_cnn(input_shape, output_shape):
    input = layers.Input(shape=input_shape)
    #branch1
    conv1 = layers.Conv2D(32, (3,3), activation='relu')(input)
    conv2 = layers.Conv2D(64, (3,3), activation='relu')(conv1)
    max_pool = layers.MaxPooling2D()(conv2)
    f1 = layers.Flatten()(max_pool)
    #branch2
    conv1 = layers.Conv2D(32, (5,5), activation='relu')(input)
    conv2 = layers.Conv2D(64, (5,5), activation='relu')(conv1)
    max_pool = layers.MaxPooling2D()(conv2)
    f2 = layers.Flatten()(max_pool)
    #branch3
    conv1 = layers.Conv2D(32, (7,7), activation='relu')(input)
    conv2 = layers.Conv2D(64, (7,7), activation='relu')(conv1)
    max_pool = layers.MaxPooling2D()(conv2)
    f3 = layers.Flatten()(max_pool)

    concat = layers.Concatenate()([f1, f2, f3])

    output = layers.Dense(output_shape, activation='softmax')(concat)

    model = models.Model(inputs=input_shape, outputs=output_shape)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Nadam',
        metrics=['accuracy']
    )

    return model




    





