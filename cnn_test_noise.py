import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
from keras.utils import to_categorical
from sys import argv

EPOCH = int(argv[2])
BATCH_SIZE = int(argv[3])

X_train = np.load(f'processed/X_train_noise.npy')
y_train = np.load(f'processed/y_train_noise.npy')
X_val = np.load(f'processed/X_val_noise.npy')
y_val = np.load(f'processed/y_val_noise.npy')


y_train = np.select([
    y_train == 'facebook',
    y_train == 'flickr',
    y_train == 'google+',
    y_train == 'instagram',
    y_train == 'original',
    y_train == 'telegram',
    y_train == 'twitter',
    y_train == 'whatsapp'
], [0,1,2,3,4,5,6,7], y_train).astype(np.uint8)


y_val = np.select([
    y_val == 'facebook',
    y_val == 'flickr',
    y_val == 'google+',
    y_val == 'instagram',
    y_val == 'original',
    y_val == 'telegram',
    y_val == 'twitter',
    y_val == 'whatsapp'
], [0,1,2,3,4,5,6,7], y_val).astype(np.uint8)


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


model = models.Sequential()
model.add(layers.Conv2D(32, 3, 3, activation='relu', input_shape=(64,64)))
model.add(layers.Conv2D(32, 3, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Conv2D(64, 3, 3, activation='relu'))
model.add(layers.Conv2D(64, 3, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Flatten())
model.add(layers.Dense(units=256))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(units=8, activation='softmax'))

model.summary()

callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)


model.compile(
    loss='categorical_crossentropy',
    optimizer='AdaDelta',
    metrics=['accuracy']
)


model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[callback], validation_data=(X_val, y_val))

model.save(f'models/cnn_noise')



