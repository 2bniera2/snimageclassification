from tensorflow.keras.utils import Sequence
import h5py
import numpy as np

class data_generator(Sequence):
    def __init__(self, X_dset, y_dset, dset_len, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.X_dset = X_dset
        self.y_dset = y_dset
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


        with h5py.File(self.X_dset) as X_dset_, h5py.File(self.y_dset) as y_dset_:
            for i in batch_indices:
                X.append(X_dset_['DCT'][i])
                y.append(self.one_hot_encode(y_dset_['labels'][i]))
                


        return np.array(X), np.array(y)






    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def one_hot_encode(self, label):
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
        one_hot_label[label_map[label.decode('UTF-8')]] = 1
        return one_hot_label

