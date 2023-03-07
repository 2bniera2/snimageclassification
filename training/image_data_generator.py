from keras.utils import Sequence
import h5py
import numpy as np

from tensorflow.image import grayscale_to_rgb

class image_data_generator(Sequence):
    def __init__(self, dset_path, dset_name, classes, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.dset_path = dset_path
        self.dset_name = dset_name
        self.dset_len = self.get_dset_len()
        self.indices = [i for i in range(self.dset_len)]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.classes = classes

    def __len__(self):
        return int(np.ceil(self.dset_len / self.batch_size))

    def get_dset_len(self):
        with h5py.File(self.dset_path, 'r') as f:
            return f[self.dset_name].shape[0]


    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size : min((index+1)*self.batch_size, len(self.indices))]
        X = []
        y = []

        with h5py.File(self.dset_path) as dset:
            for i in batch_indices:
                X.append(dset[self.dset_name][i])
                y.append(self.one_hot_encode(dset['labels'][i]))
        
      

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def one_hot_encode(self, label):
        one_hot_label = np.zeros(len(self.classes))
        one_hot_label[int(label[0])] = 1
        return one_hot_label

