from keras import layers, models


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


    





