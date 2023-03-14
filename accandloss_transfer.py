import os
import pandas as pd
import matplotlib.pyplot as plt


results = {
    'resnet502dhist' : '/home/lazy/snimageclassification/models/cnn_resnet50_(201, 62)_10_32.log',
    'vgg16dctimage' : '/home/lazy/snimageclassification/models/cnn_vgg_16_(224, 224)_10_20.log'
    
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