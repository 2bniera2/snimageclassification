import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from collections import defaultdict
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sys import argv

classes = ['facebook', 'flickr', 'google+', 'instagram', 'original', 'telegram', 'twitter', 'whatsapp']


# load test data and model and get predictions
X_test = np.load(f'processed/X_test_{argv[1]}.npy')
y_test = np.load(f'processed/y_test_{argv[1]}.npy')
model = models.load_model(f'models/cnn_{argv[1]}')
y_pred = np.argmax(model.predict(X_test), axis=1)


print(y_test)


X_test = (X_test - X_test.mean()) / (X_test.std())

# convert labels with image index to just regular labels (to test on patch level)
for i, y in enumerate(y_test): 
    y_truth = y.split('.')
    y_test[i] = y_truth[0]

# convert test labels into integers
y_test_int = np.select([
    y_test == 'facebook',
    y_test == 'flickr',
    y_test == 'google+',
    y_test == 'instagram',
    y_test == 'original',
    y_test == 'telegram',
    y_test == 'twitter',
    y_test == 'whatsapp'
], [0,1,2,3,4,5,6,7], y_test).astype(np.uint8)


# convert prediction labels into strings
y_pred_str = np.select([
    y_pred == 0,
    y_pred == 1,
    y_pred == 2,
    y_pred == 3,
    y_pred == 4,
    y_pred == 5,
    y_pred == 6,
    y_pred == 7
], [
'facebook',
'flickr',
'google+',
'instagram',
'original',
'telegram',
'twitter',
'whatsapp'
], y_pred)



print(classification_report(y_test_int, y_pred, target_names=classes))

cm = confusion_matrix(
	y_test,
	y_pred_str,
	labels=classes,
    normalize='true'
	
)


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

cm_display.plot()
plt.show()



print("### Pathces to entire images ###")



y_test = np.load(f'processed/y_test_{argv[1]}.npy')

df = pd.DataFrame([y_test, y_pred_str], index=['truth', 'prediction']).T


grouped_df = df.groupby('truth', as_index=False)['prediction'].agg(pd.Series.mode)

grouped_df['truth'] = grouped_df['truth'].str.split('.').str[0]

y_truth = grouped_df['truth'].to_numpy()
y_pred = grouped_df['prediction'].to_numpy()


print(classification_report(y_truth, y_pred))

cm = confusion_matrix(
	y_truth,
	y_pred,
	labels=classes,
    normalize='true'
	
)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

cm_display.plot()
plt.show()
