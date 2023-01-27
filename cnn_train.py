'''
Usage: python cnn_train.py {name} {EPOCH} {BATCH}
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
from keras.utils import to_categorical
from sys import argv

EPOCH = int(argv[2])
BATCH_SIZE = int(argv[3])


X_train = np.load(f'processed/X_train_{argv[1]}.npy')
y_train = np.load(f'processed/y_train_{argv[1]}.npy')

X_val = np.load(f'processed/X_val_{argv[1]}.npy')
y_val = np.load(f'processed/y_val_{argv[1]}.npy')

# X_train = (X_train - X_train.mean())/(X_train.std())
# X_val = (X_val - X_val.mean())/(X_val.std())


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

# print(len(y_train))
# print(len(y_val))

# print(len(X_train))
# print(len(X_val))


# print(y_train[0])
# print(y_val[0])


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print(len(y_train[0]))
print(len(y_val[0]))

model = models.Sequential()
model.add(layers.Conv1D(100, 3,activation='relu', input_shape=(909,1)))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(100, 3,activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation='relu', ))
model.add(layers.Dense(units=256))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(units = 8, activation='softmax'))


model.summary()

callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.compile(
    loss='categorical_crossentropy',
    optimizer='AdaDelta',
    metrics=['accuracy']
)


model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[callback], validation_data=(X_val, y_val))

model.save(f'models/cnn_{argv[1]}')
