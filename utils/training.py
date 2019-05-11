import math

from keras import backend as K
from keras.callbacks import Callback

# Custom functions
def step_decay(epoch):
    ''' Decrease learning rate by 1e-1 every 10 epochs
    '''
    initial_lrate = 1e-3
    drop = 1e-1
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop,
        math.floor((1+epoch)/epochs_drop))
    lrate = 1e-4 if lrate < 1e-4 else lrate
    return lrate


class PrintLR(Callback):
    ''' Print the learning rate
    '''
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("\033[1m==> Learning Rate: %.2e\033[0m\n" % (K.eval(lr_with_decay)))
