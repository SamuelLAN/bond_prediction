import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
keras = tf.keras


class Board(keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.model.optimizer, 'lr'):
            lr = K.get_value(self.model.optimizer.lr)
            if isinstance(lr, np.float32):
                logs.update({'lr': lr})
        super().on_epoch_end(epoch, logs)
