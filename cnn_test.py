import tensorflow as tf
import numpy as np
from tensorflow.keras import models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')





model = models.load_model('2017_cnn')

y_pred = model.predict(X_test)
