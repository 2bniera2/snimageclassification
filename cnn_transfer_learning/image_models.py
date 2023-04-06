from keras import layers, models, applications, optimizers, regularizers
from regulariser import add_regularization


def vgg_16_stock(input_shape, output_shape):
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

def vgg_16_frozen(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.VGG16(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    base.trainable = False

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

def vgg_16_reg(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.VGG16(include_top=False,input_shape=input_shape, weights='imagenet')(input)

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    model = add_regularization(model, regularizers.l2(0.001))

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

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def vgg_19_frozen(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.VGG19(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    base.trainable = False

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def vgg_19_reg(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.VGG19(include_top=False,input_shape=input_shape, weights='imagenet')(input)

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    model = add_regularization(model, regularizers.l2(0.001))

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

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model
    
def resnet50_frozen(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.ResNet50(include_top=False,input_shape=input_shape, weights='imagenet')(input)
    base.trainable = False

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def resnet50_reg(input_shape, output_shape):
    input_shape = (*input_shape, 3)
    input = layers.Input(shape=input_shape)
    base = applications.ResNet50(include_top=False,input_shape=input_shape, weights='imagenet')(input)

    flat = layers.Flatten()(base)
    output = layers.Dense(output_shape, activation="softmax")(flat)
    model = models.Model(inputs=input, outputs=output)
    model = add_regularization(model, regularizers.l2(0.001))

    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )


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

