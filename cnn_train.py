import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from keras.utils import to_categorical

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


X_train = np.load('processed/X_train.npy')
y_train = np.load('processed/y_train.npy')
x_val = np.load('processed/X_val.npy')
y_val = np.load('processed/y_val.npy')


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
model.add(layers.Conv1D(100, 3,activation='relu', input_shape=(909,1)))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(100, 3,activation='relu', input_shape=(909,1)))
model.add(layers.MaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation='relu', ))
model.add(layers.Dense(units=256))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(units = 8, activation='softmax'))


model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='AdaDelta',
    metrics=['accuracy']
)


print(f"train shape {X_train.shape} train labels {y_train.shape} val shape {x_val.shape} val labels {y_val.shape}")

print(y_val)
print(y_train)


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val))

model.save('models/2017_cnn')
