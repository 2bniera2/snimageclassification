import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from collections import defaultdict

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load test data and model and get predictions
X_test = np.load('processed/X_test.npy')
y_test = np.load('processed/y_test.npy')
model = models.load_model('models/2017_cnn')
y_pred = model.predict(X_test)



# convert class probabilities into a classified class
yhat = [y.argmax(axis = 0) for y in y_pred]

#build a structure that will store classes, each image in that class and the predictions for each patch of
#image 
classes_and_predictions = defaultdict(lambda: defaultdict(list))

for label, pred in zip(y_test, yhat):    
    y = label.split('.')
    classes_and_predictions[y[0]][y[1]].append(pred)

print(classes_and_predictions['facebook'])


# obtain an accuracy for each class
class_num = {
    0 : 'facebook',
    1 : 'flickr',
    2 : 'google+',
    3 : 'imgur',
    4 : 'instagram',
    5 : 'tinypic',
    6 : 'twitter',
    7 : 'whatsapp'
}

class_accuracies = []

for c in classes_and_predictions.items():
    sn = c[0]
    total = len(c[1])

    correct = 0
    class_predictions = c[1]

    for p in class_predictions.items():
        prediction = max(p[1], key = p[1].count)
        if class_num[prediction] == sn: correct += 1

    class_accuracies.append((sn, (correct / total) * 100))

print(class_accuracies) 


