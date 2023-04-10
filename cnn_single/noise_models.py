from keras import layers, models




def prnu_cnn(input_shape, output_shape):
    input = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3,3), activation='relu')(input)
    conv2 = layers.Conv2D(32, (3,3), activation='relu')(conv1)
    max_pool1 = layers.MaxPool2D()(conv2)
    dropout1 = layers.Dropout(rate=0.5)(max_pool1)
    conv3 = layers.Conv2D(64, (3,3), activation='relu')(dropout1)
    conv4 = layers.Conv2D(64, (3,3), activation='relu')(conv3)
    max_pool2 = layers.MaxPool2D()(conv4)
    dropout2 = layers.Dropout(rate=0.5)(max_pool2)
    flat = layers.Flatten()(dropout2)
    dense1 = layers.Dense(256, activation='relu')(flat)
    dropout3 = layers.Dropout(rate=0.5)(dense1)
    output = layers.Dense(units=output_shape, activation='softmax')(dropout3)

    model = models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model
