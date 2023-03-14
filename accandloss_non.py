import os
import pandas as pd
import matplotlib.pyplot as plt


results = {
    'dct_his:[-20,20]' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_(369, 1)_10_32.log',
    'dct_his:[-50,50]' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_(909, 1)_10_32.log',
    'dct_his:[-100,100]' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_(1809, 1)_10_32.log',
    'dct_sf:[11:44]' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_(3333, 1)_10_32.log',
    'dct_sf:sf:[1:44]' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_(4343, 1)_10_32.log',
    'dct_conv_2' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_2_(909, 1)_10_32.log',
    'dct_conv_5' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_5_(909, 1)_10_32.log',
    'dct_1000_unit_dense' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_dense_(909, 1)_10_32.log',
    'dct_0.8_dropout' : '/home/lazy/snimageclassification/models/cnn_dct_cnn_hi_dropout_(909, 1)_10_32.log',
    'noiseprint' : '/home/lazy/snimageclassification/models/cnn_prnu_cnn_(64, 64, 1)_10_32.log',
    'fusionNET' : '/home/lazy/snimageclassification/models/FusionNET.log',
}




train_accs = []
train_losses = []
val_accs = []
val_losses = []

for key, value in results.items():
    df = pd.read_csv(value)
    train_acc = df['accuracy']
    train_loss = df['loss']
    val_acc = df['val_accuracy']
    val_loss = df['val_loss']
    
    # Add to the lists for each metric
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(val_acc)
    val_losses.append(val_loss)

# Plot train accuracy and loss in separate windows
plt.figure()
for train_acc in train_accs:
    plt.plot(train_acc)
plt.title('Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(results.keys())
plt.show()

plt.figure()
for train_loss in train_losses:
    plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(results.keys())
plt.show()

# Plot validation accuracy and loss in separate windows
plt.figure()
for val_acc in val_accs:
    plt.plot(val_acc)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(results.keys())
plt.show()

plt.figure()
for val_loss in val_losses:
    plt.plot(val_loss)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(results.keys())
plt.show()