from keras import layers, models


def FusionNET(input1_shape, input2_shape, output_shape):
    print(f"{input1_shape} {input2_shape}")

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

def two_stream(input1_shape, input2_shape, output_shape):
    #dct
    input1 = layers.Input(shape=input1_shape)
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
    input2 = layers.Input(shape=input2_shape)
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




    





