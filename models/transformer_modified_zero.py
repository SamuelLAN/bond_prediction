import tensorflow as tf
from lib.nn_model_base import NN
from tf_learning_rate.warmup_then_down import CustomSchedule
from tf_models.transformer_modified_rezero import Transformer, create_padding_mask, create_look_ahead_mask
from tf_metrics.multi_label_classification import tf_accuracy, tf_precision, tf_recall, tf_f1, tf_hamming_loss

keras = tf.keras
layers = keras.layers


class Model(NN):
    params = {
        **NN.default_params,
        'learning_rate': 2e-4,  # 2e-4,
        # 'learning_rate': 1e-3,  # 2e-4,
        'emb_dim': 128,
        'dim_model': 128,
        # 'dim_model': 256,
        'ff_units': 128,
        # 'ff_units': 256,
        # 'num_layers': 12,
        'num_layers': 6,
        'num_heads': 8,
        'use_embeddings': True,
        'share_embeddings': True,
        'zero_initial': False,
        # 'lr_decay_rate': 0.01,
        # 'lr_staircase': False,
        # 'lr_factor': 0.6,
        # 'lr_patience': 10,
        'batch_size': 64,
        'epoch': 3000,
        'early_stop': 30,
        'monitor': 'val_tf_f1',
        'monitor_mode': 'max',
        'monitor_start_train': 'tf_accuracy',
        'monitor_start_train_val': 0.90,
        'dropout': 0.1,
        # 'kernel_initializer': 'glorot_uniform',
        # 'loss': 'categorical_crossentropy',
        'loss': 'mean_squared_error',
        'max_pe_input': 17,
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

    def new_mean_square_loss(self, y_true, y_pred, from_logits=False, label_smoothing=0):
        # label smoothing
        label_smoothing = 0.1
        num_classes = tf.cast(self.__num_classes, y_pred.dtype)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        return tf.reduce_mean(tf.square(y_true - y_pred) / (2 - tf.square(y_pred)))
        # K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.compat.v1.train.AdamOptimizer,
            # 'optimizer': tf.keras.optimizers.Adam,
            # 'loss': keras.losses.binary_crossentropy,
            # 'loss': keras.losses.categorical_crossentropy,
            'loss': keras.losses.mean_squared_error,
            # 'loss': self.new_mean_square_loss,
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
        # self.set_learning_rate(CustomSchedule(self.params['dim_model']))

        """ Build neural network architecture """
        self.model = Transformer(
            num_layers=self.params['num_layers'],
            d_model=self.params['dim_model'],
            num_heads=self.params['num_heads'],
            d_ff=self.params['ff_units'],
            input_vocab_size=self.__input_dim,
            target_vocab_size=self.__num_classes,
            max_pe_input=self.params['max_pe_input'],
            max_pe_target=self.params['max_pe_input'],
            drop_rate=self.params['dropout'],
            use_embeddings=self.params['use_embeddings'],
            share_embeddings=self.params['share_embeddings'],
            zero_initial=self.params['zero_initial'],
        )

    def predict(self, x):
        return self.model.predict(x, batch_size=self.params['batch_size'])

    def analyze(self):
        print('analyze ... ')

        _layers = self.model.layers
        _weights = self.model.get_weights()

        print('finish analyzing')

    def test(self, train_x, train_y_one_hot, val_x, val_y_one_hot, mask=None, name='val', data_size=None):
        return self.test_in_batch(val_x, val_y_one_hot, mask, name, data_size)
