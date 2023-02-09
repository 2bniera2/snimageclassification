from tensorflow.keras import layers, models

def dct_cnn_2017(model, input_shape):
    
    model.add(layers.Conv1D(100, 3, activation='relu', padding="valid", input_shape=input_shape))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(100, 3, activation='relu', padding="valid"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=256))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=8, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

def dct_cnn_2017_padding(model, input_shape):
    model.add(layers.Conv1D(100, 3, activation='relu', padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(100, 3, activation='relu', padding="same"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=256))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=8, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )



def dct_cnn_2019(model, input_shape):
    model.add(layers.Conv1D(100, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(100, 3, activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1000))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=1000))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=8, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

def dct_cnn_2019_padding(model, input_shape):
    model.add(layers.Conv1D(100, 3, activation='relu', padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(100, 3, activation='relu', padding="same"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1000))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=1000))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=8, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

