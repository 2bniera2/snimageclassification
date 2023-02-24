from tensorflow.keras.utils import Sequence
import h5py
import numpy as np

class data_generator(Sequence):
    def __init__(self, dset_path, dset_name, dset_len, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.dset_path = dset_path
        self.dset_name = dset_name
        self.dset_len = dset_len
        self.indices = [i for i in range(dset_len)]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dset_len / self.batch_size))

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
        one_hot_label = np.zeros(8)
        one_hot_label[int(label[0])] = 1
        return one_hot_label

