'''
Usage: python cnn_train.py {name} {EPOCH} {BATCH}
'''

from sys import argv, path 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of the tf startup messages
from tensorflow.keras import models, callbacks
from sklearn import utils
import numpy as np
import h5py
import model_architectures



# obtain dataset lengths
def get_dset_len(path, dset):
    with h5py.File(path, 'r') as f:
        return f[dset].shape[0]

def one_hot_encode(label):
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
    one_hot_label = np.zeros(len(label_map))
    one_hot_label[label_map[lable.decode('UTF-8')]] = 1
    return one_hot_label


# generator function to feed data to model in batches
def generator(batch_size, num_examples, task, name):

    # label_map = {
    #     'facebook': 0,
    #     'flickr': 1,
    #     'google+': 2,
    #     'instagram': 3,
    #     'original': 4,
    #     'telegram': 5,
    #     'twitter': 6,
    #     'whatsapp': 7
    # }
    with h5py.File(f'{path[0]}/processed/DCT_{task}_{name}.h5', 'r') as X, h5py.File(f'{path[0]}/processed/labels_{task}_{name}.h5', 'r') as y:
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
                    batch_labels.append(one_hot_encode(l))
                # shuffle
                X_batch, y_batch = utils.shuffle(
                    batch_X, np.array(batch_labels))

                yield (np.array(X_batch), np.array(y_batch))




def main(name, epoch, batch_size, architecture):

    train_dset_len = get_dset_len(f'{path[0]}/processed/DCT_train_{name}.h5', 'DCT')
    val_dset_len =  get_dset_len(f'{path[0]}/processed/DCT_val_{name}.h5', 'DCT')

    # layers
    model = models.Sequential()
    getattr(model_architectures, architecture)(model)
    
    # early stopping to prevent doing unnecessary epochs
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # train
    model.fit(
        generator(batch_size, train_dset_len, 'train', name),
        steps_per_epoch=np.ceil(train_dset_len / batch_size),
        epochs=epoch,
        callbacks=[callback],
        validation_data=(generator(batch_size, val_dset_len, 'val', name)),
        validation_steps=np.ceil(val_dset_len / batch_size),
    )

    # save
    model.save(f'models/cnn_{argv[1]}')

if __name__ == "__main__":
    main()
