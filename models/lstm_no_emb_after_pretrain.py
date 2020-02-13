import numpy as np
import tensorflow as tf
from lib.nn_model_base import NN
from models.word2vec import Model as Word2vec
from tf_metrics.multi_label_classification import tf_accuracy, tf_precision, tf_recall, tf_f1, tf_hamming_loss
from models.lstm_no_emb import Model as LSTM

keras = tf.keras
layers = keras.layers


class Model(NN):
    params = {
        **NN.default_params,
        'learning_rate': 1e-3,
        # 'lr_decay_rate': 0.01,
        # 'lr_staircase': False,
        # 'lr_factor': 0.6,
        # 'lr_patience': 10,
        'batch_size': 32,
        'epoch': 1000,
        'early_stop': 100,
        'monitor': 'val_tf_f1',
        'monitor_mode': 'max',
        'monitor_start_train': 'tf_accuracy',
        'monitor_start_train_val': 0.70,
        # 'dropout': 0.5,
        # 'kernel_initializer': 'glorot_uniform',
        'loss': 'categorical_crossentropy',
        # 'loss': 'mean_squared_error',
        # 'loss': 'mean_squared_error + categorical_crossentropy',
    }

    # @staticmethod
    def self_defined_loss(self, y_true, y_pred, from_logits=False, label_smoothing=0):
        return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing) / float(
            self.num_classes)

        # return keras.losses.mean_squared_error(y_true, y_true * y_pred + (1 - y_true) * y_pred) + \
        #        keras.losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing) * 0.002
        # return keras.losses.mean_squared_error(y_true, (1 - y_true) * y_pred)

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.train.AdamOptimizer,
            'loss': keras.losses.categorical_crossentropy,
            # 'loss': keras.losses.mean_squared_error,
            # 'loss': self.self_defined_loss,
            'metrics': [
                tf_accuracy,
                tf_hamming_loss,
                tf_f1,
                tf_precision,
                tf_recall,
            ],
            'callbacks': [
                self.callback_tf_board,
                self.callback_saver,
                # self.callback_reduce_lr,
            ],
        }

    def build(self):
        """ Build neural network architecture """

        o_lstm = LSTM('2020_01_12_18_49_46', 'lstm_for_pretrain_with_same_volume', 2007)
        o_lstm.compile(0.001)
        o_lstm.load_model(
            r'D:\Github\bond_prediction\runtime\models\lstm_for_pretrain_with_same_volume\2020_01_12_18_49_46\lstm_for_pretrain_with_same_volume.030-0.0596.hdf5',
            np.zeros([1, 20, 2007]),
            np.zeros([1, 20, 2007]))

        double_lstm_layers = o_lstm.model.layers[1:-2]
        model_layers = [
                           layers.LSTM(500, return_sequences=True),
                           layers.Dropout(0.5),
                       ] + double_lstm_layers + [
                           layers.Dropout(0.5),
                           layers.Dense(self.num_classes, activation='sigmoid'),
                       ]
        self.model = keras.Sequential(model_layers)

        # self.model = keras.Sequential([
        #     double_lstm,
        #     # layers.Bidirectional(layers.LSTM(500, return_sequences=True)),
        #     layers.LSTM(500, return_sequences=True),
        #     layers.LSTM(500),
        #     # layers.BatchNormalization(),
        #     layers.Dropout(0.5),
        #     layers.Dense(500, activation='sigmoid'),
        #     # layers.BatchNormalization(),
        #     layers.Dropout(0.5),
        #     layers.Dense(self.num_classes, activation='sigmoid'),
        # ])

# o_model = Model('test', 'lstm_test', 509)
