from keras import layers, models, applications, optimizers, regularizers
from regulariser import add_regularization


def vgg_16(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.VGG16(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    # base.trainable = False

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    # model = add_regularization(model, regularizers.l2(0.001))

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def vgg_19(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.VGG19(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    # base.trainable = False

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    # model = add_regularization(model, regularizers.l2(0.001))

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def resnet50(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.ResNet50(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    # base.trainable = False

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    # model = add_regularization(model, regularizers.l2(0.001))

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def inception(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.InceptionV3(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    # base.trainable = False

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    # model = add_regularization(model, regularizers.l2(0.001))

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
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
        optimizer='Adam',
        metrics=['accuracy']
    )

    return model
