from keras.callbacks import Callback
import time

class time_logger(Callback):
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'w') as f:
            f.write('epoch,time\n')


    def on_train_begin(self, logs=None):
        self.time_log = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_taken = time.time() - self.epoch_start_time
        self.time_log[epoch] = time_taken
        with open(self.filename, 'a') as f:
            f.write(f'{epoch},{time_taken}\n')
