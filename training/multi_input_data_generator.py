from keras.utils import Sequence
import h5py
import numpy as np


class multi_input_data_generator(Sequence):
    def __init__(self, dset1_path, dset2_path, dset1_name, dset2_name,classes, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.dset1_path = dset1_path
        self.dset2_path = dset2_path
        self.dset1_name = dset1_name
        self.dset2_name = dset2_name
        self.dset_len = self.get_dset_len()
        self.indices = [i for i in range(self.dset_len)]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.classes = classes

    def __len__(self):
        return int(np.ceil(self.dset_len / self.batch_size))

    def get_dset_len(self):
        with h5py.File(self.dset1_path, 'r') as f:
            return f[self.dset1_name].shape[0]


    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size : min((index+1)*self.batch_size, len(self.indices))]
        X1 = []
        X2 = []

        y1 = []
        y2 = []

        with h5py.File(self.dset1_path) as dset1, h5py.File(self.dset2_path) as dset2:
            for i in batch_indices:
                X1.append(dset1[self.dset1_name][i])
                y1.append(self.one_hot_encode(dset1['labels'][i]))            
                X2.append(dset2[self.dset2_name][i])
                y2.append(self.one_hot_encode(dset2['labels'][i]))
                
        return [X1,X2], [y1,y2]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def one_hot_encode(self, label):
        one_hot_label = np.zeros(len(self.classes))
        one_hot_label[int(label[0])] = 1
        return one_hot_label

