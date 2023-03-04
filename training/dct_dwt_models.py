from keras import layers, models, applications


def dct_vgg_16(input_shape, output_shape):
    input = applications.VGG16(False, "imagenet",input_shape=input_shape)
    output = layers.Dense(output_shape, activation="softmax")(input)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model


def dct_vgg_19(input_shape, output_shape):
    input = applications.VGG19(False, "imagenet",input_shape=input_shape)
    output = layers.Dense(output_shape, activation="softmax")(input)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model



def dct_res_net50(input_shape, output_shape):
    input = applications.ResNet50(False, "imagenet",input_shape=input_shape)
    output = layers.Dense(output_shape, activation="softmax")(input)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model



def dct_res_net101(input_shape, output_shape):
    input = applications.ResNet50(False, "imagenet",input_shape=input_shape)
    output = layers.Dense(output_shape, activation="softmax")(input)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model


def dct_inception(input_shape, output_shape):
    input = applications.InceptionV3(False, "imagenet",input_shape=input_shape)
    output = layers.Dense(output_shape, activation="softmax")(input)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

    return model

