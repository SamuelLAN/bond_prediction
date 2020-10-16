import tensorflow as tf
from lib.nn_model_base import NN
from tf_learning_rate.warmup_then_down import CustomSchedule
from tf_models.transformer_modified_dense import Transformer, create_padding_mask, create_look_ahead_mask
from tf_metrics.multi_label_classification import tf_accuracy, tf_precision, tf_recall, tf_f1, tf_hamming_loss

keras = tf.keras
layers = keras.layers


class Model(NN):
    params = {
        **NN.default_params,
        'learning_rate': 2e-4,  # 2e-4,
        'emb_dim': 128,
        'dim_model': 128,
        'ff_units': 128,
        'num_layers': 6,
        'num_heads': 8,
        'use_embeddings': True,
        'share_embeddings': True,
        # 'lr_decay_rate': 0.01,
        # 'lr_staircase': False,
        # 'lr_factor': 0.6,
        # 'lr_patience': 10,
        'batch_size': 128,
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
            # 'optimizer': tf.keras.optimizers.Adam,
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
        # self.set_learning_rate(CustomSchedule(self.params['dim_model']))

        """ Build neural network architecture """
        self.model = Transformer(
            num_layers=self.params['num_layers'],
            d_model=self.params['dim_model'],
            num_heads=self.params['num_heads'],
            d_ff=self.params['ff_units'],
            input_vocab_size=self.__input_dim,
            target_vocab_size=self.__num_classes,
            max_pe_input=20,
            max_pe_target=20,
            drop_rate=self.params['dropout'],
            use_embeddings=self.params['use_embeddings'],
            share_embeddings=self.params['share_embeddings'],
        )

    def predict(self, x):
        return self.model.predict(x, batch_size=self.params['batch_size'])

    def test(self, train_x, train_y_one_hot, val_x, val_y_one_hot, mask=None, name='val', data_size=None):
        return self.test_in_batch(val_x, val_y_one_hot, mask, name, data_size)
