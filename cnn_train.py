'''
Usage: python cnn_train.py {name} {EPOCH} {BATCH}
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
from tensorflow.keras import layers, models, callbacks
from sklearn import utils
import numpy as np
import h5py
from sys import argv



# obtain dataset lengths
def get_dset_len(path, dset):
    with h5py.File(path, 'r') as f:
        return f[dset].shape[0]


def model_architecture(model):
    model.add(layers.Conv1D(100, 3, activation='relu', input_shape=(909, 1)))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(100, 3, activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu', ))
    model.add(layers.Dense(units=256))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=8, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='AdaDelta',
        metrics=['accuracy']
    )

# generator function to feed data to model in batches
def generator(batch_size, num_examples, task, name):

    label_map = {
        'facebook': 0,
        'flickr': 1,
        'google+': 2,
        'instagram': 3,
        'original': 4,
        'telegram': 5,
        'twitter': 6,
        'whatsapp': 7
    }
    with h5py.File(f'processed/DCT_{task}_{name}.h5', 'r') as X, h5py.File(f'processed/labels_{task}_{name}.h5', 'r') as y:
        examples = X['DCT']
        labels = y['labels']

        while True:

            for i in range(0, num_examples, batch_size):

                # prevents exceeding bounds of list
                batch_X = examples[i: min(i + batch_size, num_examples)]
                batch_y = labels[i: min(i + batch_size, num_examples)]

                batch_labels = []
                # one hot encode labels
                for l in batch_y:
                    one_hot_label = np.zeros(len(label_map))
                    one_hot_label[label_map[l.decode('UTF-8')]] = 1
                    batch_labels.append(one_hot_label)
                # shuffle
                X_batch, y_batch = utils.shuffle(
                    batch_X, np.array(batch_labels))

                yield (np.array(X_batch), np.array(y_batch))




def main():
    # user defined variables
    name = argv[1]
    EPOCH = int(argv[2])
    BATCH_SIZE = int(argv[3])

    train_dset_len = get_dset_len(f'processed/DCT_train_{name}.h5', 'DCT')
    val_dset_len =  get_dset_len(f'processed/DCT_val_{name}.h5', 'DCT')

    # layers
    model = models.Sequential()
    model_architecture(model)

    # early stopping to prevent doing unnecessary epochs
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # train
    model.fit(
        generator(BATCH_SIZE, train_dset_len, 'train', name),
        steps_per_epoch=np.ceil(train_dset_len / BATCH_SIZE),
        epochs=EPOCH,
        callbacks=[callback],
        validation_data=(generator(BATCH_SIZE, val_dset_len, 'val', name)),
        validation_steps=np.ceil(val_dset_len / BATCH_SIZE),
    )

    # save
    model.save(f'models/cnn_{argv[1]}')

if __name__ == "__main__":
    main()
