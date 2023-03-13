import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from IPython import display


df = pd.read_csv('/home/lazy/snimageclassification/models/cnn_dct_cnn_2017_(909, 1)_10_16_report.csv')


classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter', 'whatsapp']


ax = df.iloc[:6, :4].plot(kind='bar')

# set the x-axis tick labels
ax.set_xticklabels(classes)

# show the plot
plt.show()