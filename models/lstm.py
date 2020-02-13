import tensorflow as tf
from lib.nn_model_base import NN
from models.word2vec import Model as Word2vec
from tf_metrics.multi_label_classification import tf_accuracy, tf_precision, tf_recall, tf_f1, tf_hamming_loss

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
        'batch_size': 20,
        'epoch': 3000,
        'early_stop': 300,
        'monitor': 'val_tf_hamming_loss',
        'monitor_mode': 'min',
        'monitor_start_train': 'tf_accuracy',
        'monitor_start_train_val': 0.95,
        # 'dropout': 0.5,
        # 'kernel_initializer': 'glorot_uniform',
        'loss': 'categorical_crossentropy',
        # 'loss': 'mean_squared_error',
        # 'loss': 'mean_squared_error + categorical_crossentropy',
    }

    # @staticmethod
    def self_defined_loss(self, y_true, y_pred, from_logits=False, label_smoothing=0):
        # return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing) / float(
        #     self.num_classes)

        return keras.losses.mean_squared_error(y_true, y_true * y_pred + (1 - y_true) * y_pred * 1.05) * 100. + \
               keras.losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing) / float(
            self.num_classes)
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
        self.model = keras.Sequential([
            # layers.Bidirectional(layers.LSTM(200, return_sequences=True)),
            # layers.BatchNormalization(),
            layers.LSTM(150, return_sequences=True),
            layers.LSTM(150, return_sequences=True),
            # layers.LSTM(300, return_sequences=True),
            layers.LSTM(150),
            # layers.BatchNormalization(),
            # layers.Dropout(0.5),
            # layers.Dense(200, activation='sigmoid'),
            # layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='sigmoid'),
        ])
