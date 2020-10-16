import tensorflow as tf
from lib.nn_model_base import NN
from tf_models.fc import FC
from tf_metrics.multi_label_classification import tf_accuracy, tf_precision, tf_recall, tf_f1, tf_hamming_loss


keras = tf.keras
layers = keras.layers


class Model(NN):
    params = {
        **NN.default_params,
        'learning_rate': 1e-3,
        'emb_dim': 128,
        'unit_list': [128, 128],
        'mode': 'concat',
        'use_embeddings': True,
        'batch_size': 64,
        'epoch': 3000,
        'early_stop': 30,
        'monitor': 'val_tf_f1',
        'monitor_mode': 'max',
        'monitor_start_train': 'tf_accuracy',
        'monitor_start_train_val': 0.70,
        'dropout': 0.1,
        # 'kernel_initializer': 'glorot_uniform',
        # 'loss': 'categorical_crossentropy',
        'loss': 'mean_squared_error',
        # 'loss': 'mean_squared_error + categorical_crossentropy',
    }

    def __init__(self, input_dim, model_dir, model_name=None, num_classes=None):
        self.__input_dim = input_dim
        self.__num_classes = num_classes
        super(Model, self).__init__(model_dir, model_name, num_classes)

    # @staticmethod
    def self_defined_loss(self, y_true, y_pred, from_logits=False, label_smoothing=0):
        return keras.losses.categorical_crossentropy(y_true, y_pred, True, 0.1)
        # return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing) / float(self.num_classes)

        # return keras.losses.mean_squared_error(y_true, y_true * y_pred + (1 - y_true) * y_pred) + \
        #        keras.losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing) * 0.002
        # return keras.losses.mean_squared_error(y_true, (1 - y_true) * y_pred)

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.compat.v1.train.AdamOptimizer,
            # 'loss': keras.losses.binary_crossentropy,
            # 'loss': keras.losses.categorical_crossentropy,
            'loss': keras.losses.mean_squared_error,
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
        self.model = FC(
            input_dim=self.__input_dim,
            emb_dim=self.params['emb_dim'],
            unit_list=self.params['unit_list'] + [self.__num_classes],
            dropout_rate=self.params['dropout'],
            activation='tanh',
            mode=self.params['mode'],
            use_embeddings=self.params['use_embeddings'],
        )
