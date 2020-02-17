from keras.callbacks import Callback
from keras import backend as K

class AlphaScheduling(Callback):
    def __init__(self, alpha, scheduling):
        self.alpha = alpha
        self.scheduling = scheduling

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.alpha, self.scheduling(epoch))
