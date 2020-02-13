import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from load.interval_input_output import Loader

keras = tf.keras
layers = keras.layers

o_load = Loader()
train_x, train_y = o_load.train()
test_x, test_y = o_load.test()

# _mean = np.mean(train_x, axis=0)
# _std = np.std(train_x, axis=0)
# train_x = (train_x - _mean) / (_std + 0.0001)
# test_x = (test_x - _mean) / (_std + 0.0001)

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(2000, activation='sigmoid'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1995, activation='sigmoid'),
])

default_params = {
    'learning_rate': 1e-5,
    'lr_decay_rate': 0.01,
    'lr_staircase': False,
    'lr_factor': 0.1,
    'lr_patience': 3,
    'batch_size': 32,
    'epoch': 1200,
    'early_stop': 20,
    'auto_open_tensorboard': True,
    'monitor': 'val_categorical_accuracy',
    'monitor_mode': 'max',
    'monitor_start_train_acc': 0.5,
    'initial_epoch': 0,
    'random_state': 42,
}


def tf_accuracy(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))


def tf_precision(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    # y_not_true = tf.cast(tf.equal(y_true, 0), tf.int32)

    true_positive = tf.reduce_sum(y_true * y_pred)
    # false_positive = tf.reduce_sum(y_not_true * y_pred)
    predict_true_num = tf.reduce_sum(y_pred)
    return true_positive / predict_true_num
    # return true_positive / (true_positive + false_positive)


def tf_recall(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    true_positive = tf.reduce_sum(y_true * y_pred)
    y_true_num = tf.reduce_sum(y_true)
    return true_positive / y_true_num


def tf_f1(y_true, y_pred):
    precision = tf_precision(y_true, y_pred)
    recall = tf_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def tf_hamming_loss(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    y_not_pred = tf.cast(tf.equal(y_pred, 0), tf.int32)
    y_not_true = tf.cast(tf.equal(y_true, 0), tf.int32)
    return tf.reduce_mean(tf.cast(y_not_true * y_pred, tf.float32) + tf.cast(y_true * y_not_pred, tf.float32))


def accuracy(y_true, y_pred):
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return np.mean(y_true == y_pred)


# model param config
params = {**default_params}

config_for_keras = {
    'optimizer': tf.train.AdamOptimizer,
    'loss': keras.losses.categorical_crossentropy,
    'metrics': [
        tf_accuracy,
        tf_precision,
        tf_recall,
        tf_f1,
        tf_hamming_loss,
    ],
    'callbacks': [
    ],
}

model.compile(optimizer=config_for_keras['optimizer'](learning_rate=params['learning_rate']),
              loss=config_for_keras['loss'],
              metrics=config_for_keras['metrics'])

model.fit(train_x, train_y,
          epochs=params['epoch'],
          validation_data=[test_x, test_y],
          batch_size=params['batch_size'],
          verbose=2)

train_y_pred = model.predict(train_x)
_acc = accuracy(train_y, train_y_pred)

print('-------------------------')
print(_acc)

test_y_pred = model.predict(test_x)
_acc = accuracy(test_y, test_y_pred)

print('-------------------------')
print(_acc)
