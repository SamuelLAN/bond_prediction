import tensorflow as tf


def tf_accuracy(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0), tf.int32)
    y_true = tf.cast(tf.greater_equal(y_true, 0), tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))


def tf_precision(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0), tf.int32)
    y_true = tf.cast(tf.greater_equal(y_true, 0), tf.int32)
    true_positive = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
    predict_true_num = tf.cast(tf.reduce_sum(y_pred), tf.float32)
    return true_positive / (predict_true_num + 0.0001)


def tf_recall(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0), tf.int32)
    y_true = tf.cast(tf.greater_equal(y_true, 0), tf.int32)
    true_positive = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
    y_true_num = tf.cast(tf.reduce_sum(y_true), tf.float32)
    return true_positive / (y_true_num + 0.0001)


def tf_f1(y_true, y_pred):
    precision = tf_precision(y_true, y_pred)
    recall = tf_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + 0.0001)


def tf_hamming_loss(y_true, y_pred):
    y_pred = tf.cast(tf.greater_equal(y_pred, 0), tf.int32)
    y_true = tf.cast(tf.greater_equal(y_true, 0), tf.int32)
    y_not_pred = tf.cast(tf.equal(y_pred, 0), tf.int32)
    y_not_true = tf.cast(tf.equal(y_true, 0), tf.int32)
    return tf.reduce_mean(tf.cast(y_not_true * y_pred, tf.float32) + tf.cast(y_true * y_not_pred, tf.float32))
