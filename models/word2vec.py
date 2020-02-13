import numpy as np
import tensorflow as tf
from lib.nn_model_base import NN

keras = tf.keras
layers = keras.layers


class Model(NN):
    VOC_SIZE = 6345
    EMB_SIZE = 100
    NEG_SAMPLE_NUM = 64
    WINDOW = 3

    params = {
        **NN.default_params,
        'learning_rate': 1e-6,
        'batch_size': 256,
        'epoch': 2,
        'early_stop': 30,
        'monitor': 'val_loss',
        'monitor_mode': 'min',
        # 'monitor_start_train': 'acc',
        # 'monitor_start_train_val': 0.7,
        'update_freq': 10000,
    }

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.train.AdamOptimizer,
            'loss': self.__loss,
            'metrics': [
                'accuracy',
            ],
            'callbacks': [
                self.callback_tf_board,
                self.callback_saver,
                # self.callback_reduce_lr,
            ],
        }

    def __loss(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.VOC_SIZE)
        return keras.losses.categorical_crossentropy(y_true, y_pred)

    def build(self):
        self.model = keras.Sequential([
            layers.Embedding(self.VOC_SIZE, self.EMB_SIZE,
                             # input_length=2 * self.WINDOW,
                             name='emb'),
            # layers.Flatten(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.VOC_SIZE),
        ])

    def emb_layer(self):
        return self.gen_model_from_layer(0, 0)

    def emb_matrix(self):
        return self.model.weights[0]

# o_model = Model('test', 'test')
